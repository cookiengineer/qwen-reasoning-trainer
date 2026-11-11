package inference

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"
)

const (
	markerThinkOpen  = "<think>"
	markerThinkClose = "</think>"
	markerToolOpen   = "<tool_call>"
	markerToolClose  = "</tool_call>"
)

type parserMode int

const (
	modeContent parserMode = iota
	modeThinking
	modeTool
)

type streamParser struct {
	mode     parserMode
	buf      string
	emit     func(Delta) error
	argTypes map[string]map[string]string

	content   strings.Builder
	reasoning strings.Builder
	toolCalls []ToolCall
}

func newStreamParser(thinking bool, argTypes map[string]map[string]string, emit func(Delta) error) *streamParser {
	mode := modeContent
	if thinking {
		mode = modeThinking
	}
	return &streamParser{mode: mode, emit: emit, argTypes: argTypes}
}

// Feed consumes decoded text and emits complete thinking/content/tool-call
// deltas. Markers split across calls are buffered until they are complete.
func (p *streamParser) Feed(s string) error {
	if s == "" {
		return nil
	}
	p.buf += s
	return p.drain(false)
}

// Close flushes any remaining text. An incomplete tool-call block is dropped.
func (p *streamParser) Close() error { return p.drain(true) }

func (p *streamParser) Result() Result {
	return Result{
		Content:          p.content.String(),
		ReasoningContent: p.reasoning.String(),
		ToolCalls:        append([]ToolCall(nil), p.toolCalls...),
	}
}

func (p *streamParser) drain(final bool) error {
	for {
		if p.mode == modeTool {
			idx := strings.Index(p.buf, markerToolClose)
			if idx < 0 {
				if final {
					p.buf = ""
				}
				return nil
			}
			block := p.buf[:idx]
			p.buf = p.buf[idx+len(markerToolClose):]
			calls := parseToolCalls(block, p.argTypes)
			if len(calls) > 0 {
				p.toolCalls = append(p.toolCalls, calls...)
				if p.emit != nil {
					if err := p.emit(Delta{ToolCalls: calls}); err != nil {
						return err
					}
				}
			}
			p.mode = modeContent
			continue
		}

		markers := []string{markerThinkOpen, markerThinkClose, markerToolOpen}
		idx, marker := earliestMarker(p.buf, markers)
		if idx >= 0 {
			if idx > 0 {
				if err := p.emitText(p.buf[:idx]); err != nil {
					return err
				}
			}
			p.buf = p.buf[idx+len(marker):]
			switch marker {
			case markerThinkOpen:
				p.mode = modeThinking
			case markerThinkClose:
				p.mode = modeContent
			case markerToolOpen:
				p.mode = modeTool
			}
			continue
		}

		if final {
			if p.buf != "" {
				if err := p.emitText(p.buf); err != nil {
					return err
				}
				p.buf = ""
			}
			return nil
		}

		hold := partialMarkerSuffix(p.buf, markers)
		if emitLen := len(p.buf) - hold; emitLen > 0 {
			if err := p.emitText(p.buf[:emitLen]); err != nil {
				return err
			}
			p.buf = p.buf[emitLen:]
		}
		return nil
	}
}

func (p *streamParser) emitText(text string) error {
	if text == "" {
		return nil
	}
	if p.mode == modeThinking {
		p.reasoning.WriteString(text)
		if p.emit != nil {
			return p.emit(Delta{ReasoningContent: text})
		}
		return nil
	}
	p.content.WriteString(text)
	if p.emit != nil {
		return p.emit(Delta{Content: text})
	}
	return nil
}

func earliestMarker(s string, markers []string) (int, string) {
	best := -1
	bestMarker := ""
	for _, m := range markers {
		if i := strings.Index(s, m); i >= 0 && (best < 0 || i < best) {
			best = i
			bestMarker = m
		}
	}
	return best, bestMarker
}

func partialMarkerSuffix(s string, markers []string) int {
	best := 0
	for _, m := range markers {
		max := len(m) - 1
		if len(s) < max {
			max = len(s)
		}
		for l := max; l > best; l-- {
			if strings.HasSuffix(s, m[:l]) {
				best = l
				break
			}
		}
	}
	return best
}

var toolCallCounter uint64

func newToolCallID() string {
	n := atomic.AddUint64(&toolCallCounter, 1)
	return fmt.Sprintf("call_%x%x", time.Now().UnixNano(), n)
}

type argPair struct {
	key string
	val any
}

func parseToolCalls(block string, argTypes map[string]map[string]string) []ToolCall {
	var calls []ToolCall
	rest := block
	for {
		i := strings.Index(rest, "<function=")
		if i < 0 {
			break
		}
		rest = rest[i+len("<function="):]
		j := strings.IndexByte(rest, '>')
		if j < 0 {
			break
		}
		name := strings.TrimSpace(rest[:j])
		rest = rest[j+1:]
		end := strings.Index(rest, "</function>")
		var body string
		if end < 0 {
			body = rest
			rest = ""
		} else {
			body = rest[:end]
			rest = rest[end+len("</function>"):]
		}
		if name == "" {
			continue
		}
		args := parseParameters(body, argTypes[name])
		calls = append(calls, ToolCall{
			ID:   newToolCallID(),
			Type: "function",
			Function: FunctionCall{
				Name:      name,
				Arguments: args,
			},
		})
		if rest == "" {
			break
		}
	}
	return calls
}

func parseParameters(body string, types map[string]string) string {
	var pairs []argPair
	seen := map[string]bool{}
	for {
		i := strings.Index(body, "<parameter=")
		if i < 0 {
			break
		}
		body = body[i+len("<parameter="):]
		j := strings.IndexByte(body, '>')
		if j < 0 {
			break
		}
		key := strings.TrimSpace(body[:j])
		body = body[j+1:]
		end := strings.Index(body, "</parameter>")
		var raw string
		if end < 0 {
			raw = body
			body = ""
		} else {
			raw = body[:end]
			body = body[end+len("</parameter>"):]
		}
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		pairs = append(pairs, argPair{key: key, val: coerceValue(trimParamValue(raw), types[key])})
	}
	return marshalArgs(pairs)
}

func trimParamValue(s string) string {
	s = strings.TrimPrefix(s, "\r\n")
	s = strings.TrimPrefix(s, "\n")
	s = strings.TrimSuffix(s, "\r\n")
	s = strings.TrimSuffix(s, "\n")
	return s
}

func coerceValue(raw, typ string) any {
	s := strings.TrimSpace(raw)
	switch typ {
	case "number":
		var n json.Number
		if err := json.Unmarshal([]byte(s), &n); err == nil {
			return n
		}
	case "integer":
		var i int64
		if err := json.Unmarshal([]byte(s), &i); err == nil {
			return i
		}
	case "boolean":
		var b bool
		if err := json.Unmarshal([]byte(s), &b); err == nil {
			return b
		}
	case "object", "array":
		var v any
		if err := json.Unmarshal([]byte(s), &v); err == nil {
			return v
		}
	}
	return raw
}

func marshalArgs(pairs []argPair) string {
	if len(pairs) == 0 {
		return "{}"
	}
	var b strings.Builder
	b.WriteByte('{')
	for i, p := range pairs {
		if i > 0 {
			b.WriteByte(',')
		}
		key, _ := json.Marshal(p.key)
		val, err := json.Marshal(p.val)
		if err != nil {
			val, _ = json.Marshal(fmt.Sprint(p.val))
		}
		b.Write(key)
		b.WriteByte(':')
		b.Write(val)
	}
	b.WriteByte('}')
	return b.String()
}
