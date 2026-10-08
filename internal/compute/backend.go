package compute

// Capabilities describes a backend's device limits and supported features.
type Capabilities struct {
	Name          string
	FP16          bool
	BF16          bool
	MaxBufferSize uint64
}

// Buffer is an opaque handle to device-resident tensor storage.
type Buffer interface {
	Dims() []int
	NumElements() int
}

// Backend abstracts a compute device. The CPU reference backend and the Vulkan
// backend implement it. Higher-level graph execution will build on this.
type Backend interface {
	// Capabilities reports device features and limits.
	Capabilities() Capabilities
	// Upload copies a host tensor into device memory.
	Upload(t *Tensor) (Buffer, error)
	// Download copies device memory back into a new host tensor.
	Download(b Buffer) (*Tensor, error)
	// Free releases a device buffer.
	Free(b Buffer)
	// Sync blocks until all submitted work completes.
	Sync() error
}
