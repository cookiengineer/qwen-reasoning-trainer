package quant

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// This file is a float32-faithful port of llama.cpp's IQ3_S quantizer
// (ggml-quants.c: iq3xs_init_impl + quantize_row_iq3_s_impl + the _ref wrapper),
// so internal/quant's IQ3_S encoder is byte-identical to the reference. The
// 512-entry grid table lives in iq3s_kgrid.go.

const (
	iq3SBlockSize = 32
	iq3SMaxQ      = 8
	iq3SGridSize  = 512
	iq3SKMapSize  = 4096
	iq3SNWant     = 3
)

var (
	iq3Once       sync.Once
	iq3Grid       [][4]int8
	iq3Map        []int32
	iq3Neighbours []uint16
)

// iq3Init builds the grid, map and neighbour tables exactly as
// iq3xs_init_impl(512) does. It runs once.
func iq3Init() {
	grid := make([][4]int8, iq3SGridSize)
	for k := 0; k < iq3SGridSize; k++ {
		for i := 0; i < 4; i++ {
			l := (iq3KGrid512[k] >> (3 * i)) & 0x7
			grid[k][i] = int8(2*l + 1)
		}
	}
	kmap := make([]int32, iq3SKMapSize)
	for i := range kmap {
		kmap[i] = -1
	}
	for i := 0; i < iq3SGridSize; i++ {
		var index uint16
		for k := 0; k < 4; k++ {
			q := uint16((int(grid[i][k]) - 1) / 2)
			index |= q << (3 * k)
		}
		kmap[index] = int32(i)
	}
	nPerI := make([]int, iq3SKMapSize)
	numNeighbors, numNotInMap := 0, 0
	for i := 0; i < iq3SKMapSize; i++ {
		if kmap[i] >= 0 {
			continue
		}
		numNotInMap++
		pos := iq3Pos(i)
		order := iq3SortOrder(grid, pos)
		n, d2, nhave := 0, order[0].d2, 1
		for j := 0; j < iq3SGridSize; j++ {
			if order[j].d2 > d2 {
				if nhave == iq3SNWant {
					break
				}
				d2 = order[j].d2
				nhave++
			}
			n++
		}
		nPerI[i] = n
		numNeighbors += n
	}
	neighbours := make([]uint16, numNeighbors+numNotInMap)
	offsets := make([]int, iq3SKMapSize)
	counter := 0
	for i := 0; i < iq3SKMapSize; i++ {
		if kmap[i] >= 0 {
			offsets[i] = -1
			continue
		}
		offsets[i] = counter
		counter += 1 + nPerI[i]
	}
	for i := 0; i < iq3SKMapSize; i++ {
		if kmap[i] >= 0 {
			continue
		}
		order := iq3SortOrder(grid, iq3Pos(i))
		lc := offsets[i]
		kmap[i] = int32(-(lc + 1))
		d2 := order[0].d2
		start := lc
		lc++
		n, nhave := 0, 1
		for j := 0; j < iq3SGridSize; j++ {
			if order[j].d2 > d2 {
				if nhave == iq3SNWant {
					break
				}
				d2 = order[j].d2
				nhave++
			}
			neighbours[lc] = uint16(order[j].idx)
			lc++
			n++
		}
		neighbours[start] = uint16(n)
	}
	iq3Grid, iq3Map, iq3Neighbours = grid, kmap, neighbours
}

type iq3Dist struct{ d2, idx int }

func iq3Pos(i int) [4]int8 {
	var pos [4]int8
	for k := 0; k < 4; k++ {
		l := (i >> (3 * k)) & 0x7
		pos[k] = int8(2*l + 1)
	}
	return pos
}

func iq3SortOrder(grid [][4]int8, pos [4]int8) []iq3Dist {
	order := make([]iq3Dist, iq3SGridSize)
	for j := 0; j < iq3SGridSize; j++ {
		d2 := 0
		for k := 0; k < 4; k++ {
			d := int(grid[j][k]) - int(pos[k])
			d2 += d * d
		}
		order[j] = iq3Dist{d2, j}
	}
	// Same total order as iq3_compare_func: by d2 then index.
	sort.Slice(order, func(a, b int) bool {
		if order[a].d2 != order[b].d2 {
			return order[a].d2 < order[b].d2
		}
		return order[a].idx < order[b].idx
	})
	return order
}

// iq3FindBestNeighbour ports iq3_find_best_neighbour. u is the map index
// (kmap_q3xs offset); L[0:4] receives the grid coordinates.
func iq3FindBestNeighbour(u int, xval, weight []float32, scale float32, L []int8) int32 {
	base := -int(iq3Map[u]) - 1
	numNeighbors := int(iq3Neighbours[base])
	bestD2 := float32(math.MaxFloat32)
	gridIndex := -1
	for j := 1; j <= numNeighbors; j++ {
		gi := int(iq3Neighbours[base+j])
		pg := iq3Grid[gi]
		var d2 float32
		for i := 0; i < 4; i++ {
			q := float32(pg[i])
			diff := scale*q - xval[i]
			d2 += weight[i] * diff * diff
		}
		if d2 < bestD2 {
			bestD2 = d2
			gridIndex = gi
		}
	}
	pg := iq3Grid[gridIndex]
	for i := 0; i < 4; i++ {
		L[i] = int8((int(pg[i]) - 1) / 2)
	}
	return int32(gridIndex)
}

// quantizeIQ3S encodes src into IQ3_S blocks (110 bytes / 256 elements).
func quantizeIQ3S(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: iq3_s requires a multiple of %d elements", QK_K)
	}
	iq3Once.Do(iq3Init)
	nbl := len(src) / QK_K
	const bs4 = iq3SBlockSize / 4
	const bs8 = iq3SBlockSize / 8
	var scales [QK_K / iq3SBlockSize]float32
	var weight, xval, waux [iq3SBlockSize]float32
	var L, Laux [iq3SBlockSize]int8
	var isOnGrid, isOnGridAux [iq3SBlockSize / 4]bool
	var blockSigns [iq3SBlockSize / 8]uint8

	for ibl := 0; ibl < nbl; ibl++ {
		block := src[ibl*QK_K : (ibl+1)*QK_K]
		b := dst[ibl*110:]
		for j := range b {
			b[j] = 0
		}
		qs := b[2:66]
		qh := b[66:74]
		signs := b[74:106]
		scOut := b[106:110]

		var maxScale float32

		qsOff, signsOff := 0, 0
		for ib := 0; ib < QK_K/iq3SBlockSize; ib++ {
			xb := block[ib*iq3SBlockSize:]
			for i := 0; i < iq3SBlockSize; i++ {
				weight[i] = xb[i] * xb[i]
			}
			for i := 0; i < iq3SBlockSize; i++ {
				waux[i] = sqrtf(weight[i])
			}
			for k := 0; k < bs8; k++ {
				var s uint8
				for i := 0; i < 8; i++ {
					if xb[8*k+i] >= 0 {
						xval[8*k+i] = xb[8*k+i]
					} else {
						xval[8*k+i] = -xb[8*k+i]
						s |= 1 << uint(i)
					}
				}
				blockSigns[k] = s
			}
			maxV := xval[0]
			for i := 1; i < iq3SBlockSize; i++ {
				if xval[i] > maxV {
					maxV = xval[i]
				}
			}
			for i := range L {
				L[i] = 0
			}
			if maxV == 0 {
				scales[ib] = 0
				continue
			}
			var best float32
			scale := maxV / (2*iq3SMaxQ - 1)
			for k := 0; k < bs4; k++ {
				isOnGrid[k] = false
			}
			for is := -9; is <= 9; is++ {
				id := (2*iq3SMaxQ - 1 + float32(is)*0.2) / maxV
				thisScale := 1 / id
				for k := 0; k < bs4; k++ {
					for i := 0; i < 4; i++ {
						l := nearestInt(0.5 * (id*xval[4*k+i] - 1))
						Laux[4*k+i] = int8(clampInt(l, 0, iq3SMaxQ-1))
					}
					var u uint16
					for i := 0; i < 4; i++ {
						u |= uint16(Laux[4*k+i]) << (3 * i)
					}
					gridIndex := iq3Map[u]
					isOnGridAux[k] = true
					if gridIndex < 0 {
						isOnGridAux[k] = false
						gridIndex = iq3FindBestNeighbour(int(u), xval[4*k:], waux[4*k:], thisScale, Laux[4*k:])
					}
				}
				var sumqx, sumq2 float32
				for i := 0; i < iq3SBlockSize; i++ {
					w := weight[i]
					q := 2*float32(Laux[i]) + 1
					sumqx += w * xval[i] * q
					sumq2 += w * q * q
				}
				if sumq2 > 0 && sumqx*sumqx > best*sumq2 {
					scale = sumqx / sumq2
					best = scale * sumqx
					L = Laux
					for k := 0; k < bs4; k++ {
						isOnGrid[k] = isOnGridAux[k]
					}
				}
			}
			nNotOnGrid := 0
			for k := 0; k < bs4; k++ {
				if !isOnGrid[k] {
					nNotOnGrid++
				}
			}
			if nNotOnGrid > 0 && scale > 0 {
				id := 1 / scale
				for k := 0; k < bs4; k++ {
					var u uint16
					for i := 0; i < 4; i++ {
						l := nearestInt(0.5 * (id*xval[4*k+i] - 1))
						l = clampInt(l, 0, iq3SMaxQ-1)
						u |= uint16(l) << (3 * i)
					}
					gridIndex := iq3Map[u]
					if gridIndex < 0 {
						gridIndex = iq3FindBestNeighbour(int(u), xval[4*k:], waux[4*k:], scale, L[4*k:])
					}
					pg := iq3Grid[gridIndex]
					for i := 0; i < 4; i++ {
						L[4*k+i] = int8((int(pg[i]) - 1) / 2)
					}
				}
				var sumqx, sumq2 float32
				for i := 0; i < iq3SBlockSize; i++ {
					w := weight[i]
					q := 2*float32(L[i]) + 1
					sumqx += w * xval[i] * q
					sumq2 += w * q * q
				}
				if sumq2 > 0 {
					scale = sumqx / sumq2
				}
			}
			if scale < 0 {
				scale = -scale
				for k := 0; k < bs8; k++ {
					blockSigns[k] = ^blockSigns[k]
				}
			}
			for k := 0; k < bs4; k++ {
				var u uint16
				for i := 0; i < 4; i++ {
					u |= uint16(L[4*k+i]) << (3 * i)
				}
				gridIndex := iq3Map[u]
				if gridIndex < 0 {
					return fmt.Errorf("quant: iq3_s point not on grid")
				}
				qs[qsOff+k] = byte(gridIndex & 0xFF)
				qh[(ib*bs4+k)/8] |= byte(gridIndex>>8) << uint((ib*bs4+k)%8)
			}
			qsOff += bs4
			for k := 0; k < bs8; k++ {
				signs[signsOff+k] = blockSigns[k]
			}
			signsOff += bs8
			scales[ib] = scale
			if scale > maxScale {
				maxScale = scale
			}
		}

		if maxScale == 0 {
			continue
		}
		d := maxScale / 31
		putHalf(b[0:2], d*1.033)
		id := 1 / d
		for ib := 0; ib < QK_K/iq3SBlockSize; ib += 2 {
			l1 := clampInt(nearestInt(0.5*(id*scales[ib+0]-1)), 0, 15)
			l2 := clampInt(nearestInt(0.5*(id*scales[ib+1]-1)), 0, 15)
			scOut[ib/2] = byte(l1 | (l2 << 4))
		}
	}
	return nil
}
