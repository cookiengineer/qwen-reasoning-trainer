// Package vulkan implements compute.Backend on top of Vulkan 1.1 compute
// pipelines. It loads the Vulkan loader dynamically with purego (no cgo) and
// uses precompiled SPIR-V shaders embedded in the binary.
//
// Memory is allocated as host-visible, host-coherent device memory so buffers
// can be mapped directly; this favors simplicity and correctness over peak
// throughput. Complex ops without a shader yet (SSM conv, GatedDeltaNet) are
// executed on the host as a fallback.
package vulkan

import (
	"embed"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

//go:embed spirv/*.spv
var shaderFS embed.FS

// maxBindings is the number of storage-buffer bindings in the shared
// descriptor set layout. Shaders use a subset; the atomic GatedDeltaNet
// backward uses fifteen.
const maxBindings = 16

// Available reports whether a Vulkan loader can be opened.
func Available() bool {
	lib, err := purego.Dlopen("libvulkan.so.1", purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		lib, err = purego.Dlopen("libvulkan.so", purego.RTLD_NOW|purego.RTLD_LOCAL)
	}
	if err != nil {
		return false
	}
	purego.Dlclose(lib)
	return true
}

// Vulkan constants used by this backend.
const (
	vkSuccess = 0

	vkStructureApplicationInfo        = 0
	vkStructureInstanceCreateInfo     = 1
	vkStructureDeviceQueueCreateInfo  = 2
	vkStructureDeviceCreateInfo       = 3
	vkStructureSubmitInfo             = 4
	vkStructureMemoryAllocateInfo     = 5
	vkStructureFenceCreateInfo        = 8
	vkStructureBufferCreateInfo       = 12
	vkStructureShaderModuleCreateInfo = 16
	vkStructurePipelineShaderStageCi  = 18
	vkStructureComputePipelineCi      = 29
	vkStructurePipelineLayoutCi       = 30
	vkStructureDescriptorSetLayoutCi  = 32
	vkStructureDescriptorPoolCi       = 33
	vkStructureDescriptorSetAi        = 34
	vkStructureWriteDescriptorSet     = 35
	vkStructureCommandPoolCi          = 39
	vkStructureCommandBufferAi        = 40
	vkStructureCommandBufferBeginInfo = 42
	vkStructureMemoryBarrier          = 46

	// Vulkan 1.1 / extension struct type tags.
	vkStructurePhysicalDeviceFeatures2                    = 1000059000
	vkStructurePhysicalDeviceShaderAtomicFloatFeaturesEXT = 1000260000

	vkBufferUsageTransferSrc     = 0x00000001
	vkBufferUsageTransferDst     = 0x00000002
	vkBufferUsageStorage         = 0x00000020
	vkSharingModeExclusive       = 0
	vkMemoryPropertyDevice       = 0x00000001
	vkMemoryPropertyHostVisible  = 0x00000002
	vkMemoryPropertyHostCoherent = 0x00000004
	vkQueueComputeBit            = 0x00000002
	vkDescriptorTypeStorage      = 7
	vkShaderStageComputeBit      = 0x00000020
	vkPipelineBindPointCompute   = 1
	vkCommandBufferLevelPrimary  = 0
	vkCommandBufferUsageOneTime  = 0x00000001
	vkAccessShaderRead           = 0x00000020
	vkAccessShaderWrite          = 0x00000040
	vkPipelineStageCompute       = 0x00000800
	vkPipelineStageTransfer      = 0x00001000
	vkAccessTransferRead         = 0x00000200
	vkAccessTransferWrite        = 0x00000800
	vkDescriptorPoolFreeSet      = 0x00000001
	vkApiVersion10               = 0x00400000
	vkApiVersion11               = 0x00401000
)

// ---- Vulkan structs ----

type applicationInfo struct {
	sType              uint32
	pNext              uintptr
	pApplicationName   uintptr
	applicationVersion uint32
	pEngineName        uintptr
	engineVersion      uint32
	apiVersion         uint32
}

type instanceCreateInfo struct {
	sType                   uint32
	pNext                   uintptr
	flags                   uint32
	pApplicationInfo        uintptr
	enabledLayerCount       uint32
	ppEnabledLayerNames     uintptr
	enabledExtensionCount   uint32
	ppEnabledExtensionNames uintptr
}

type deviceQueueCreateInfo struct {
	sType            uint32
	pNext            uintptr
	flags            uint32
	queueFamilyIndex uint32
	queueCount       uint32
	pQueuePriorities uintptr
}

type deviceCreateInfo struct {
	sType                   uint32
	pNext                   uintptr
	flags                   uint32
	queueCreateInfoCount    uint32
	pQueueCreateInfos       uintptr
	enabledLayerCount       uint32
	ppEnabledLayerNames     uintptr
	enabledExtensionCount   uint32
	ppEnabledExtensionNames uintptr
	pEnabledFeatures        uintptr
}

type memoryAllocateInfo struct {
	sType           uint32
	pNext           uintptr
	allocationSize  uint64
	memoryTypeIndex uint32
}

// extensionProperties mirrors VkExtensionProperties; extensionName is
// VK_MAX_EXTENSION_NAME_SIZE (256) bytes.
type extensionProperties struct {
	extensionName [256]byte
	specVersion   uint32
}

// physicalDeviceFeatures2 mirrors VkPhysicalDeviceFeatures2 with enough room for
// the core VkPhysicalDeviceFeatures (55 bools).
type physicalDeviceFeatures2 struct {
	sType    uint32
	_        uint32
	pNext    uintptr
	features [64]uint32
}

// physicalDeviceShaderAtomicFloatFeatures mirrors
// VkPhysicalDeviceShaderAtomicFloatFeaturesEXT.
type physicalDeviceShaderAtomicFloatFeatures struct {
	sType                           uint32
	_                               uint32
	pNext                           uintptr
	shaderBufferFloat32Atomics      uint32
	shaderBufferFloat32AtomicAdd    uint32
	shaderSharedFloat32Atomics      uint32
	shaderSharedFloat32AtomicAdd    uint32
	shaderBufferFloat16Atomics      uint32
	shaderBufferFloat16AtomicAdd    uint32
	shaderBufferFloat16AtomicMinMax uint32
	shaderBufferFloat32AtomicMinMax uint32
	shaderSharedFloat16Atomics      uint32
	shaderSharedFloat16AtomicAdd    uint32
	shaderSharedFloat16AtomicMinMax uint32
	shaderSharedFloat32AtomicMinMax uint32
	shaderImageFloat32Atomics       uint32
	shaderImageFloat32AtomicAdd     uint32
	sparseImageFloat32Atomics       uint32
	sparseImageFloat32AtomicAdd     uint32
}

type bufferCreateInfo struct {
	sType                 uint32
	pNext                 uintptr
	flags                 uint32
	size                  uint64
	usage                 uint32
	sharingMode           uint32
	queueFamilyIndexCount uint32
	pQueueFamilyIndices   uintptr
}

type memoryRequirements struct {
	size           uint64
	alignment      uint64
	memoryTypeBits uint32
}

type memoryType struct {
	propertyFlags uint32
	heapIndex     uint32
}

type memoryHeap struct {
	size  uint64
	flags uint32
}

type physicalDeviceMemoryProperties struct {
	memoryTypeCount uint32
	memoryTypes     [32]memoryType
	memoryHeapCount uint32
	memoryHeaps     [16]memoryHeap
}

type queueFamilyProperties struct {
	queueFlags                  uint32
	queueCount                  uint32
	timestampValidBits          uint32
	minImageTransferGranularity [3]uint32
}

type physicalDeviceProperties struct {
	apiVersion        uint32
	driverVersion     uint32
	vendorID          uint32
	deviceID          uint32
	deviceType        uint32
	deviceName        [256]byte
	pipelineCacheUUID [16]byte
	rest              [1536]byte
}

type shaderModuleCreateInfo struct {
	sType    uint32
	pNext    uintptr
	flags    uint32
	codeSize uintptr
	pCode    uintptr
}

type descriptorSetLayoutBinding struct {
	binding         uint32
	descriptorType  uint32
	descriptorCount uint32
	stageFlags      uint32
	pImmutable      uintptr
}

type descriptorSetLayoutCreateInfo struct {
	sType        uint32
	pNext        uintptr
	flags        uint32
	bindingCount uint32
	pBindings    uintptr
}

type pushConstantRange struct {
	stageFlags uint32
	offset     uint32
	size       uint32
}

type pipelineLayoutCreateInfo struct {
	sType                  uint32
	pNext                  uintptr
	flags                  uint32
	setLayoutCount         uint32
	pSetLayouts            uintptr
	pushConstantRangeCount uint32
	pPushConstantRanges    uintptr
}

type pipelineShaderStageCreateInfo struct {
	sType               uint32
	pNext               uintptr
	flags               uint32
	stage               uint32
	module              uintptr
	pName               uintptr
	pSpecializationInfo uintptr
}

type computePipelineCreateInfo struct {
	sType              uint32
	pNext              uintptr
	flags              uint32
	stage              pipelineShaderStageCreateInfo
	layout             uintptr
	basePipelineHandle uintptr
	basePipelineIndex  int32
}

type descriptorPoolSize struct {
	typ             uint32
	descriptorCount uint32
}

type descriptorPoolCreateInfo struct {
	sType         uint32
	pNext         uintptr
	flags         uint32
	maxSets       uint32
	poolSizeCount uint32
	pPoolSizes    uintptr
}

type descriptorSetAllocateInfo struct {
	sType              uint32
	pNext              uintptr
	descriptorPool     uintptr
	descriptorSetCount uint32
	pSetLayouts        uintptr
}

type descriptorBufferInfo struct {
	buffer uintptr
	offset uint64
	rng    uint64
}

type writeDescriptorSet struct {
	sType            uint32
	pNext            uintptr
	dstSet           uintptr
	dstBinding       uint32
	dstArrayElement  uint32
	descriptorCount  uint32
	descriptorType   uint32
	pImageInfo       uintptr
	pBufferInfo      uintptr
	pTexelBufferView uintptr
}

type commandPoolCreateInfo struct {
	sType            uint32
	pNext            uintptr
	flags            uint32
	queueFamilyIndex uint32
}

type commandBufferAllocateInfo struct {
	sType              uint32
	pNext              uintptr
	commandPool        uintptr
	level              uint32
	commandBufferCount uint32
}

type commandBufferBeginInfo struct {
	sType            uint32
	pNext            uintptr
	flags            uint32
	pInheritanceInfo uintptr
}

type memoryBarrier struct {
	sType         uint32
	pNext         uintptr
	srcAccessMask uint32
	dstAccessMask uint32
}

type bufferCopy struct {
	srcOffset uint64
	dstOffset uint64
	size      uint64
}

type submitInfo struct {
	sType                uint32
	pNext                uintptr
	waitSemaphoreCount   uint32
	pWaitSemaphores      uintptr
	pWaitDstStageMask    uintptr
	commandBufferCount   uint32
	pCommandBuffers      uintptr
	signalSemaphoreCount uint32
	pSignalSemaphores    uintptr
}

type fenceCreateInfo struct {
	sType uint32
	pNext uintptr
	flags uint32
}

// vk holds resolved Vulkan entry points.
type vk struct {
	getInstanceProcAddr uintptr

	CreateInstance                     uintptr
	EnumeratePhysicalDevices           uintptr
	GetPhysicalDeviceQueueFamilyProps  uintptr
	GetPhysicalDeviceMemoryProps       uintptr
	GetPhysicalDeviceProperties        uintptr
	CreateDevice                       uintptr
	GetDeviceQueue                     uintptr
	EnumerateDeviceExtensionProperties uintptr
	GetPhysicalDeviceFeatures2         uintptr
	CreateBuffer                       uintptr
	GetBufferMemoryRequirements        uintptr
	AllocateMemory                     uintptr
	BindBufferMemory                   uintptr
	MapMemory                          uintptr
	UnmapMemory                        uintptr
	DestroyBuffer                      uintptr
	FreeMemory                         uintptr
	CreateShaderModule                 uintptr
	CreateDescriptorSetLayout          uintptr
	CreatePipelineLayout               uintptr
	CreateComputePipelines             uintptr
	CreateDescriptorPool               uintptr
	AllocateDescriptorSets             uintptr
	ResetDescriptorPool                uintptr
	UpdateDescriptorSets               uintptr
	CreateCommandPool                  uintptr
	AllocateCommandBuffers             uintptr
	ResetCommandBuffer                 uintptr
	BeginCommandBuffer                 uintptr
	CmdBindPipeline                    uintptr
	CmdBindDescriptorSets              uintptr
	CmdPushConstants                   uintptr
	CmdDispatch                        uintptr
	CmdPipelineBarrier                 uintptr
	CmdCopyBuffer                      uintptr
	EndCommandBuffer                   uintptr
	QueueSubmit                        uintptr
	QueueWaitIdle                      uintptr
	CreateFence                        uintptr
	WaitForFences                      uintptr
	ResetFences                        uintptr
	DeviceWaitIdle                     uintptr
	DestroyDevice                      uintptr
	DestroyInstance                    uintptr
}

func cstr(s string) (uintptr, []byte) {
	b := append([]byte(s), 0)
	return uintptr(unsafe.Pointer(&b[0])), b
}

func (v *vk) load(instance uintptr, name string) uintptr {
	p, kb := cstr(name)
	r, _, _ := purego.SyscallN(v.getInstanceProcAddr, instance, p)
	runtime.KeepAlive(kb)
	return r
}

func (v *vk) loadAll(instance uintptr) {
	names := map[*uintptr]string{
		&v.CreateInstance:                     "vkCreateInstance",
		&v.EnumeratePhysicalDevices:           "vkEnumeratePhysicalDevices",
		&v.GetPhysicalDeviceQueueFamilyProps:  "vkGetPhysicalDeviceQueueFamilyProperties",
		&v.GetPhysicalDeviceMemoryProps:       "vkGetPhysicalDeviceMemoryProperties",
		&v.GetPhysicalDeviceProperties:        "vkGetPhysicalDeviceProperties",
		&v.CreateDevice:                       "vkCreateDevice",
		&v.GetDeviceQueue:                     "vkGetDeviceQueue",
		&v.EnumerateDeviceExtensionProperties: "vkEnumerateDeviceExtensionProperties",
		&v.GetPhysicalDeviceFeatures2:         "vkGetPhysicalDeviceFeatures2",
		&v.CreateBuffer:                       "vkCreateBuffer",
		&v.GetBufferMemoryRequirements:        "vkGetBufferMemoryRequirements",
		&v.AllocateMemory:                     "vkAllocateMemory",
		&v.BindBufferMemory:                   "vkBindBufferMemory",
		&v.MapMemory:                          "vkMapMemory",
		&v.UnmapMemory:                        "vkUnmapMemory",
		&v.DestroyBuffer:                      "vkDestroyBuffer",
		&v.FreeMemory:                         "vkFreeMemory",
		&v.CreateShaderModule:                 "vkCreateShaderModule",
		&v.CreateDescriptorSetLayout:          "vkCreateDescriptorSetLayout",
		&v.CreatePipelineLayout:               "vkCreatePipelineLayout",
		&v.CreateComputePipelines:             "vkCreateComputePipelines",
		&v.CreateDescriptorPool:               "vkCreateDescriptorPool",
		&v.AllocateDescriptorSets:             "vkAllocateDescriptorSets",
		&v.ResetDescriptorPool:                "vkResetDescriptorPool",
		&v.UpdateDescriptorSets:               "vkUpdateDescriptorSets",
		&v.CreateCommandPool:                  "vkCreateCommandPool",
		&v.AllocateCommandBuffers:             "vkAllocateCommandBuffers",
		&v.ResetCommandBuffer:                 "vkResetCommandBuffer",
		&v.BeginCommandBuffer:                 "vkBeginCommandBuffer",
		&v.CmdBindPipeline:                    "vkCmdBindPipeline",
		&v.CmdBindDescriptorSets:              "vkCmdBindDescriptorSets",
		&v.CmdPushConstants:                   "vkCmdPushConstants",
		&v.CmdDispatch:                        "vkCmdDispatch",
		&v.CmdPipelineBarrier:                 "vkCmdPipelineBarrier",
		&v.CmdCopyBuffer:                      "vkCmdCopyBuffer",
		&v.EndCommandBuffer:                   "vkEndCommandBuffer",
		&v.QueueSubmit:                        "vkQueueSubmit",
		&v.QueueWaitIdle:                      "vkQueueWaitIdle",
		&v.CreateFence:                        "vkCreateFence",
		&v.WaitForFences:                      "vkWaitForFences",
		&v.ResetFences:                        "vkResetFences",
		&v.DeviceWaitIdle:                     "vkDeviceWaitIdle",
		&v.DestroyDevice:                      "vkDestroyDevice",
		&v.DestroyInstance:                    "vkDestroyInstance",
	}
	for ptr, n := range names {
		*ptr = v.load(instance, n)
	}
	// GetPhysicalDeviceFeatures2 is core in Vulkan 1.1; fall back to the KHR
	// alias for 1.0 instances.
	if v.GetPhysicalDeviceFeatures2 == 0 {
		v.GetPhysicalDeviceFeatures2 = v.load(instance, "vkGetPhysicalDeviceFeatures2KHR")
	}
}

// hasDeviceExtension reports whether the physical device exposes a named
// extension.
func hasDeviceExtension(v *vk, pd uintptr, name string) bool {
	if v.EnumerateDeviceExtensionProperties == 0 {
		return false
	}
	var count uint32
	if vkCall(v.EnumerateDeviceExtensionProperties, pd, 0, uintptr(unsafe.Pointer(&count)), 0) != vkSuccess || count == 0 {
		return false
	}
	props := make([]extensionProperties, count)
	if vkCall(v.EnumerateDeviceExtensionProperties, pd, 0, uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&props[0]))) != vkSuccess {
		return false
	}
	for i := range props {
		n := props[i].extensionName[:]
		end := 0
		for end < len(n) && n[end] != 0 {
			end++
		}
		if string(n[:end]) == name {
			return true
		}
	}
	return false
}

// Backend is a Vulkan compute backend.
type Backend struct {
	vk             *vk
	lib            uintptr
	instance       uintptr
	physicalDevice uintptr
	device         uintptr
	queue          uintptr
	queueFamily    uint32
	cmdPool        uintptr
	cmdBuffer      uintptr
	fence          uintptr
	descPool       uintptr
	descLayout     uintptr
	pipeLayout     uintptr
	dummy          *buffer
	memProps       physicalDeviceMemoryProperties
	caps           compute.Capabilities

	// candidates lists every compute-capable physical device seen at creation,
	// in enumeration order, for diagnostics (Backend.DeviceList).
	candidates []deviceCandidate

	mu        sync.Mutex
	pipelines map[string]uintptr
	closed    bool

	// maxScratchFloats bounds the float32 dequant scratch. It is overridable
	// in tests to exercise the row-blocking path.
	maxScratchFloats int

	// noFusedFwd / noFusedT disable the fused dequantize+GEMM paths (forward /
	// transposed) so tests can exercise the row-blocked float32 scratch
	// fallback.
	noFusedFwd bool
	noFusedT   bool

	// noPool disables buffer recycling (every Free destroys the buffer). Tests
	// only.
	noPool bool

	// forceStaging makes allocation prefer a device-local buffer that is not
	// host-visible, exercising the staging transfer path. Tests only.
	forceStaging bool

	// recording is true while compute commands are being recorded into the
	// command buffer but not yet submitted. pending holds buffers freed while
	// recording; they are destroyed after the next flush.
	recording bool
	pending   []*buffer

	// pool recycles device buffers by (size, usage, hostOnly) so a resident
	// graph does not pay vkCreateBuffer/vkAllocateMemory per op. poolBytes is
	// bounded by poolCap.
	pool        map[poolKey][]*buffer
	poolBytes   uint64
	deviceBytes uint64
	peakBytes   uint64

	// scope bounds the lifetime of temporary buffers. Buffers allocated while
	// a scope is active are logged; EndScope frees the log tail except the
	// retained buffers. Used to reclaim per-layer recompute temporaries.
	scopeActive int
	scopeMarks  []int
	scopeLog    []*buffer

	stats Stats
}

// Stats counts device activity for profiling the host-mediated vs resident
// execution paths.
type Stats struct {
	Dispatches      uint64
	Submits         uint64
	Allocs          uint64
	PoolHits        uint64
	BytesUploaded   uint64
	BytesDownloaded uint64
	PooledBytes     uint64
	DeviceBytes     uint64
	PeakDeviceBytes uint64
}

type poolKey struct {
	size    uint64
	usage   uint32
	staging bool
}

// poolCap bounds the bytes retained by the buffer pool.
const poolCap = 1 << 30

type buffer struct {
	b      *Backend
	buf    uintptr
	mem    uintptr
	mapped unsafe.Pointer
	// hostVisible reports whether the memory is mapped and can be accessed
	// directly from the host. Device-local buffers on discrete GPUs are not,
	// and require staging copies.
	hostVisible bool
	usage       uint32
	staging     bool
	// shared marks a non-owning view produced by Reshape; Free is a no-op and
	// the underlying buffer is owned by the original allocation.
	shared bool
	// scopeIdx is the buffer's slot in the active allocation scope log, or -1
	// when it is not tracked. Free clears the slot so EndScope does not free a
	// buffer the owning op already released.
	scopeIdx int
	size     uint64
	dims     []int
	typ      quant.Type
}

func (x *buffer) Dims() []int { return x.dims }
func (x *buffer) NumElements() int {
	n := 1
	for _, d := range x.dims {
		n *= d
	}
	return n
}
func (x *buffer) Type() quant.Type { return x.typ }

// New creates and initializes a Vulkan backend on the first suitable device.
func New() (*Backend, error) {
	lib, err := purego.Dlopen("libvulkan.so.1", purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		lib, err = purego.Dlopen("libvulkan.so", purego.RTLD_NOW|purego.RTLD_LOCAL)
	}
	if err != nil {
		return nil, fmt.Errorf("vulkan: load loader: %w", err)
	}

	v := &vk{}
	gipaName, kb := cstr("vkGetInstanceProcAddr")
	gipa, err := purego.Dlsym(lib, "vkGetInstanceProcAddr")
	runtime.KeepAlive(kb)
	if err != nil || gipa == 0 {
		purego.Dlclose(lib)
		return nil, fmt.Errorf("vulkan: vkGetInstanceProcAddr: %w", err)
	}
	v.getInstanceProcAddr = gipa
	_ = gipaName
	v.loadAll(0)

	var appNameBuf = append([]byte("qwen-trainer"), 0)
	app := applicationInfo{
		sType:              vkStructureApplicationInfo,
		pApplicationName:   uintptr(unsafe.Pointer(&appNameBuf[0])),
		applicationVersion: 1,
		apiVersion:         vkApiVersion11,
	}
	ici := instanceCreateInfo{
		sType:            vkStructureInstanceCreateInfo,
		pApplicationInfo: uintptr(unsafe.Pointer(&app)),
	}
	var instance uintptr
	if res := vkCall(v.CreateInstance, uintptr(unsafe.Pointer(&ici)), 0, uintptr(unsafe.Pointer(&instance))); res != vkSuccess {
		purego.Dlclose(lib)
		return nil, fmt.Errorf("vulkan: vkCreateInstance failed (%d)", int32(res))
	}
	v.loadAll(instance)

	b := &Backend{vk: v, lib: lib, instance: instance, pipelines: map[string]uintptr{}}

	var devCount uint32
	if res := vkCall(v.EnumeratePhysicalDevices, instance, uintptr(unsafe.Pointer(&devCount)), 0); res != vkSuccess {
		b.Close()
		return nil, fmt.Errorf("vulkan: enumerate devices failed")
	}
	if devCount == 0 {
		b.Close()
		return nil, errors.New("vulkan: no physical devices")
	}
	devs := make([]uintptr, devCount)
	if res := vkCall(v.EnumeratePhysicalDevices, instance, uintptr(unsafe.Pointer(&devCount)), uintptr(unsafe.Pointer(&devs[0]))); res != vkSuccess {
		b.Close()
		return nil, fmt.Errorf("vulkan: enumerate devices (2) failed")
	}

	// Enumerate compute-capable devices with their device-local memory so a
	// sensible default can be chosen on multi-GPU machines (the largest), and an
	// explicit override is honored via QWEN38_VK_DEVICE (index or name
	// substring).
	var cands []deviceCandidate
	for _, pd := range devs {
		qf, ok := pickComputeQueue(v, pd)
		if !ok {
			continue
		}
		var props physicalDeviceProperties
		vkCall(v.GetPhysicalDeviceProperties, pd, uintptr(unsafe.Pointer(&props)))
		nm := strings.TrimRight(string(props.deviceName[:]), "\x00")
		var mp physicalDeviceMemoryProperties
		vkCall(v.GetPhysicalDeviceMemoryProps, pd, uintptr(unsafe.Pointer(&mp)))
		cands = append(cands, deviceCandidate{pd: pd, qf: qf, name: nm, typ: props.deviceType, mem: deviceLocalSize(mp)})
	}
	if len(cands) == 0 {
		b.Close()
		return nil, errors.New("vulkan: no device with a compute queue")
	}
	b.candidates = cands
	chosen, err := selectDeviceCandidate(cands, os.Getenv("QWEN38_VK_DEVICE"))
	if err != nil {
		b.Close()
		return nil, err
	}
	b.physicalDevice = chosen.pd
	b.queueFamily = chosen.qf
	name := chosen.name

	vkCall(v.GetPhysicalDeviceMemoryProps, b.physicalDevice, uintptr(unsafe.Pointer(&b.memProps)))

	// Optional: VK_EXT_shader_atomic_float, used by the atomic GatedDeltaNet
	// backward. Query the feature and enable it at device creation when present.
	atomicFeatures := new(physicalDeviceShaderAtomicFloatFeatures)
	atomicFeatures.sType = vkStructurePhysicalDeviceShaderAtomicFloatFeaturesEXT
	enableAtomics := false
	if v.GetPhysicalDeviceFeatures2 != 0 && hasDeviceExtension(v, b.physicalDevice, "VK_EXT_shader_atomic_float") {
		f2 := new(physicalDeviceFeatures2)
		f2.sType = vkStructurePhysicalDeviceFeatures2
		f2.pNext = uintptr(unsafe.Pointer(atomicFeatures))
		vkCall(v.GetPhysicalDeviceFeatures2, b.physicalDevice, uintptr(unsafe.Pointer(f2)))
		enableAtomics = atomicFeatures.shaderBufferFloat32AtomicAdd != 0
		runtime.KeepAlive(f2)
	}

	prio := float32(1.0)
	qci := deviceQueueCreateInfo{
		sType:            vkStructureDeviceQueueCreateInfo,
		queueFamilyIndex: b.queueFamily,
		queueCount:       1,
		pQueuePriorities: uintptr(unsafe.Pointer(&prio)),
	}
	dci := deviceCreateInfo{
		sType:                vkStructureDeviceCreateInfo,
		queueCreateInfoCount: 1,
		pQueueCreateInfos:    uintptr(unsafe.Pointer(&qci)),
	}
	extNameBytes := append([]byte("VK_EXT_shader_atomic_float"), 0)
	var extNames [1]uintptr
	if enableAtomics {
		atomicFeatures.shaderBufferFloat32AtomicAdd = 1
		dci.pNext = uintptr(unsafe.Pointer(atomicFeatures))
		extNames[0] = uintptr(unsafe.Pointer(&extNameBytes[0]))
		dci.enabledExtensionCount = 1
		dci.ppEnabledExtensionNames = uintptr(unsafe.Pointer(&extNames[0]))
	}
	var device uintptr
	if res := vkCall(v.CreateDevice, b.physicalDevice, uintptr(unsafe.Pointer(&dci)), 0, uintptr(unsafe.Pointer(&device))); res != vkSuccess {
		b.Close()
		return nil, fmt.Errorf("vulkan: vkCreateDevice failed (%d)", int32(res))
	}
	runtime.KeepAlive(extNameBytes)
	runtime.KeepAlive(atomicFeatures)
	b.device = device
	vkCall(v.GetDeviceQueue, device, uintptr(b.queueFamily), 0, uintptr(unsafe.Pointer(&b.queue)))

	cpi := commandPoolCreateInfo{sType: vkStructureCommandPoolCi, flags: 0, queueFamilyIndex: b.queueFamily}
	if res := vkCall(v.CreateCommandPool, device, uintptr(unsafe.Pointer(&cpi)), 0, uintptr(unsafe.Pointer(&b.cmdPool))); res != vkSuccess {
		b.Close()
		return nil, fmt.Errorf("vulkan: create command pool failed")
	}
	cbai := commandBufferAllocateInfo{
		sType:              vkStructureCommandBufferAi,
		commandPool:        b.cmdPool,
		level:              vkCommandBufferLevelPrimary,
		commandBufferCount: 1,
	}
	if res := vkCall(v.AllocateCommandBuffers, device, uintptr(unsafe.Pointer(&cbai)), uintptr(unsafe.Pointer(&b.cmdBuffer))); res != vkSuccess {
		b.Close()
		return nil, fmt.Errorf("vulkan: allocate command buffer failed")
	}
	fci := fenceCreateInfo{sType: vkStructureFenceCreateInfo, flags: 1}
	if res := vkCall(v.CreateFence, device, uintptr(unsafe.Pointer(&fci)), 0, uintptr(unsafe.Pointer(&b.fence))); res != vkSuccess {
		b.Close()
		return nil, fmt.Errorf("vulkan: create fence failed")
	}

	if err := b.setupDescriptors(); err != nil {
		b.Close()
		return nil, err
	}

	b.caps = compute.Capabilities{
		Name:           "vulkan(" + name + ")",
		Float32Atomics: enableAtomics,
		MaxBufferSize:  math.MaxUint32,
		MemoryBytes:    b.deviceMemorySize(),
	}

	d, err := b.allocBuffer(16, vkBufferUsageStorage|vkBufferUsageTransferSrc|vkBufferUsageTransferDst)
	if err != nil {
		b.Close()
		return nil, err
	}
	b.dummy = d
	return b, nil
}

// deviceCandidate is a compute-capable physical device with the metadata used
// for selection.
type deviceCandidate struct {
	pd   uintptr
	qf   uint32
	name string
	typ  uint32
	mem  uint64
}

// Physical device type values (VkPhysicalDeviceType).
const (
	deviceTypeOther         = 0
	deviceTypeIntegratedGPU = 1
	deviceTypeDiscreteGPU   = 2
	deviceTypeVirtualGPU    = 3
	deviceTypeCPU           = 4
)

// deviceTypeName maps a VkPhysicalDeviceType to a short label.
func deviceTypeName(t uint32) string {
	switch t {
	case deviceTypeIntegratedGPU:
		return "integrated"
	case deviceTypeDiscreteGPU:
		return "discrete"
	case deviceTypeVirtualGPU:
		return "virtual"
	case deviceTypeCPU:
		return "cpu"
	default:
		return "other"
	}
}

// deviceRank orders device types for default selection: discrete first, then
// integrated, virtual, other, and CPU (software) last.
func deviceRank(t uint32) int {
	switch t {
	case deviceTypeDiscreteGPU:
		return 0
	case deviceTypeIntegratedGPU:
		return 1
	case deviceTypeVirtualGPU:
		return 2
	case deviceTypeCPU:
		return 4
	default:
		return 3
	}
}

// selectDeviceCandidate picks a device from candidates. spec (from
// QWEN38_VK_DEVICE) may be an index, a case-insensitive name substring, or
// empty. The default prefers a non-CPU device by type rank, then the largest
// device-local memory, so a discrete GPU wins over an iGPU and a software
// rasterizer.
func selectDeviceCandidate(cands []deviceCandidate, spec string) (deviceCandidate, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		best := cands[0]
		for _, c := range cands[1:] {
			if deviceRank(c.typ) < deviceRank(best.typ) ||
				(deviceRank(c.typ) == deviceRank(best.typ) && c.mem > best.mem) {
				best = c
			}
		}
		return best, nil
	}
	if idx, err := strconv.Atoi(spec); err == nil {
		if idx < 0 || idx >= len(cands) {
			return deviceCandidate{}, fmt.Errorf("vulkan: QWEN38_VK_DEVICE index %d out of range (0..%d)", idx, len(cands)-1)
		}
		return cands[idx], nil
	}
	lwant := strings.ToLower(spec)
	for _, c := range cands {
		if strings.Contains(strings.ToLower(c.name), lwant) {
			return c, nil
		}
	}
	return deviceCandidate{}, fmt.Errorf("vulkan: no device matching QWEN38_VK_DEVICE=%q", spec)
}

// DeviceInfo describes one compute-capable physical device.
type DeviceInfo struct {
	Index       int
	Name        string
	Type        string
	MemoryBytes uint64
	Selected    bool
}

// DeviceList returns the physical devices seen at creation, marking the one in
// use. It is used by `gpu-info` for multi-GPU diagnostics.
func (b *Backend) DeviceList() []DeviceInfo {
	out := make([]DeviceInfo, len(b.candidates))
	for i, c := range b.candidates {
		out[i] = DeviceInfo{
			Index:       i,
			Name:        c.name,
			Type:        deviceTypeName(c.typ),
			MemoryBytes: c.mem,
			Selected:    c.pd == b.physicalDevice,
		}
	}
	return out
}

// deviceLocalSize returns the total device-local heap size from memory
// properties, falling back to the largest heap when none is marked local.
func deviceLocalSize(mp physicalDeviceMemoryProperties) uint64 {
	var deviceLocal, max uint64
	for i := uint32(0); i < mp.memoryHeapCount && i < 16; i++ {
		h := mp.memoryHeaps[i]
		if h.flags&vkMemoryPropertyDevice != 0 {
			deviceLocal += h.size
		}
		if h.size > max {
			max = h.size
		}
	}
	if deviceLocal > 0 {
		return deviceLocal
	}
	return max
}

func pickComputeQueue(v *vk, pd uintptr) (uint32, bool) {
	var count uint32
	vkCall(v.GetPhysicalDeviceQueueFamilyProps, pd, uintptr(unsafe.Pointer(&count)), 0)
	if count == 0 {
		return 0, false
	}
	props := make([]queueFamilyProperties, count)
	vkCall(v.GetPhysicalDeviceQueueFamilyProps, pd, uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&props[0])))
	for i, p := range props {
		if p.queueFlags&vkQueueComputeBit != 0 && p.queueCount > 0 {
			return uint32(i), true
		}
	}
	return 0, false
}

func (b *Backend) setupDescriptors() error {
	bindings := [maxBindings]descriptorSetLayoutBinding{}
	for i := range bindings {
		bindings[i] = descriptorSetLayoutBinding{
			binding:         uint32(i),
			descriptorType:  vkDescriptorTypeStorage,
			descriptorCount: 1,
			stageFlags:      vkShaderStageComputeBit,
		}
	}
	dslci := descriptorSetLayoutCreateInfo{
		sType:        vkStructureDescriptorSetLayoutCi,
		bindingCount: uint32(len(bindings)),
		pBindings:    uintptr(unsafe.Pointer(&bindings[0])),
	}
	if res := vkCall(b.vk.CreateDescriptorSetLayout, b.device, uintptr(unsafe.Pointer(&dslci)), 0, uintptr(unsafe.Pointer(&b.descLayout))); res != vkSuccess {
		return fmt.Errorf("vulkan: create descriptor set layout failed")
	}

	rangeInfo := pushConstantRange{stageFlags: vkShaderStageComputeBit, offset: 0, size: 128}
	plci := pipelineLayoutCreateInfo{
		sType:                  vkStructurePipelineLayoutCi,
		setLayoutCount:         1,
		pSetLayouts:            uintptr(unsafe.Pointer(&b.descLayout)),
		pushConstantRangeCount: 1,
		pPushConstantRanges:    uintptr(unsafe.Pointer(&rangeInfo)),
	}
	if res := vkCall(b.vk.CreatePipelineLayout, b.device, uintptr(unsafe.Pointer(&plci)), 0, uintptr(unsafe.Pointer(&b.pipeLayout))); res != vkSuccess {
		return fmt.Errorf("vulkan: create pipeline layout failed")
	}

	poolSize := descriptorPoolSize{typ: vkDescriptorTypeStorage, descriptorCount: maxBindings * 65536}
	dpci := descriptorPoolCreateInfo{
		sType:         vkStructureDescriptorPoolCi,
		flags:         vkDescriptorPoolFreeSet,
		maxSets:       65536,
		poolSizeCount: 1,
		pPoolSizes:    uintptr(unsafe.Pointer(&poolSize)),
	}
	if res := vkCall(b.vk.CreateDescriptorPool, b.device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&b.descPool))); res != vkSuccess {
		return fmt.Errorf("vulkan: create descriptor pool failed")
	}
	return nil
}

func (b *Backend) allocBuffer(size uint64, usage uint32) (*buffer, error) {
	return b.allocBufferOpt(size, usage, false)
}

// allocHostBuffer allocates a host-visible, host-coherent buffer used as a
// staging area for transfers to and from device-local memory.
func (b *Backend) allocHostBuffer(size uint64, usage uint32) (*buffer, error) {
	return b.allocBufferOpt(size, usage, true)
}

// allocBufferOpt creates a buffer. Unless hostOnly is set, it prefers
// device-local memory (mappable when the device exposes a device-local
// host-visible type, as on integrated GPUs) and falls back to any host-visible
// coherent type. On discrete GPUs, where the large device-local heap is not
// host-visible, the returned buffer is not mapped and transfers must go through
// allocHostBuffer staging.
func (b *Backend) allocBufferOpt(size uint64, usage uint32, hostOnly bool) (*buffer, error) {
	if size == 0 {
		size = 4
	}
	if buf, ok := b.poolGet(poolKey{size: size, usage: usage, staging: hostOnly}); ok {
		buf.dims = nil
		buf.typ = quant.TypeF32
		buf.scopeIdx = -1
		b.trackAlloc(buf)
		return buf, nil
	}
	b.stats.Allocs++
	bci := bufferCreateInfo{
		sType:       vkStructureBufferCreateInfo,
		size:        size,
		usage:       usage,
		sharingMode: vkSharingModeExclusive,
	}
	var buf uintptr
	if res := vkCall(b.vk.CreateBuffer, b.device, uintptr(unsafe.Pointer(&bci)), 0, uintptr(unsafe.Pointer(&buf))); res != vkSuccess {
		return nil, fmt.Errorf("vulkan: create buffer failed")
	}
	var req memoryRequirements
	vkCall(b.vk.GetBufferMemoryRequirements, b.device, buf, uintptr(unsafe.Pointer(&req)))

	var idx uint32
	var ok bool
	if hostOnly {
		idx, ok = b.memoryType(req.memoryTypeBits, vkMemoryPropertyHostVisible|vkMemoryPropertyHostCoherent)
	} else if b.forceStaging {
		idx, ok = b.memoryTypeExclude(req.memoryTypeBits, vkMemoryPropertyDevice, vkMemoryPropertyHostVisible)
		if !ok {
			idx, ok = b.memoryType(req.memoryTypeBits, vkMemoryPropertyDevice)
		}
	} else {
		// Prefer device-local mappable; then device-local (staging); then any
		// host-visible coherent.
		idx, ok = b.memoryType(req.memoryTypeBits, vkMemoryPropertyDevice|vkMemoryPropertyHostVisible|vkMemoryPropertyHostCoherent)
		if !ok {
			idx, ok = b.memoryType(req.memoryTypeBits, vkMemoryPropertyDevice)
		}
		if !ok {
			idx, ok = b.memoryType(req.memoryTypeBits, vkMemoryPropertyHostVisible|vkMemoryPropertyHostCoherent)
		}
	}
	if !ok {
		vkCall(b.vk.DestroyBuffer, b.device, buf, 0)
		return nil, errors.New("vulkan: no suitable memory type")
	}
	mai := memoryAllocateInfo{sType: vkStructureMemoryAllocateInfo, allocationSize: req.size, memoryTypeIndex: idx}
	var mem uintptr
	if res := vkCall(b.vk.AllocateMemory, b.device, uintptr(unsafe.Pointer(&mai)), 0, uintptr(unsafe.Pointer(&mem))); res != vkSuccess {
		vkCall(b.vk.DestroyBuffer, b.device, buf, 0)
		return nil, fmt.Errorf("vulkan: allocate memory failed")
	}
	if res := vkCall(b.vk.BindBufferMemory, b.device, buf, mem, 0); res != vkSuccess {
		vkCall(b.vk.FreeMemory, b.device, mem, 0)
		vkCall(b.vk.DestroyBuffer, b.device, buf, 0)
		return nil, fmt.Errorf("vulkan: bind buffer memory failed")
	}
	hostVisible := b.memProps.memoryTypes[idx].propertyFlags&vkMemoryPropertyHostVisible != 0
	var mapped unsafe.Pointer
	if hostVisible {
		if res := vkCall(b.vk.MapMemory, b.device, mem, 0, uintptr(size), 0, uintptr(unsafe.Pointer(&mapped))); res != vkSuccess {
			vkCall(b.vk.FreeMemory, b.device, mem, 0)
			vkCall(b.vk.DestroyBuffer, b.device, buf, 0)
			return nil, fmt.Errorf("vulkan: map memory failed")
		}
	}
	b.deviceBytes += req.size
	if b.deviceBytes > b.peakBytes {
		b.peakBytes = b.deviceBytes
	}
	out := &buffer{b: b, buf: buf, mem: mem, mapped: mapped, hostVisible: hostVisible, usage: usage, staging: hostOnly, scopeIdx: -1, size: size, typ: quant.TypeF32}
	b.trackAlloc(out)
	return out, nil
}

// trackAlloc records a buffer allocated while a scope is active.
func (b *Backend) trackAlloc(x *buffer) {
	if b.scopeActive > 0 {
		x.scopeIdx = len(b.scopeLog)
		b.scopeLog = append(b.scopeLog, x)
	}
}

// BeginScope implements compute.Backend.
func (b *Backend) BeginScope() {
	b.scopeMarks = append(b.scopeMarks, len(b.scopeLog))
	b.scopeActive++
}

// EndScope implements compute.Backend. It frees every buffer allocated since
// the matching BeginScope except those in keep. Buffers freed while recording
// are released on the next flush; callers must Sync before relying on the
// memory being reclaimed.
func (b *Backend) EndScope(keep ...compute.Buffer) {
	if b.scopeActive == 0 {
		return
	}
	b.scopeActive--
	mark := b.scopeMarks[len(b.scopeMarks)-1]
	b.scopeMarks = b.scopeMarks[:len(b.scopeMarks)-1]
	keepSet := make(map[*buffer]bool, len(keep))
	for _, k := range keep {
		if kb, ok := k.(*buffer); ok {
			keepSet[kb] = true
		}
	}
	victims := append([]*buffer(nil), b.scopeLog[mark:]...)
	b.scopeLog = b.scopeLog[:mark]
	for _, x := range victims {
		if x == nil {
			continue
		}
		if x.scopeIdx >= mark {
			x.scopeIdx = -1
		}
		if !keepSet[x] {
			b.Free(x)
		}
	}
}

// poolGet returns a recycled buffer of the given key, if any.
func (b *Backend) poolGet(key poolKey) (*buffer, bool) {
	list := b.pool[key]
	if len(list) == 0 {
		return nil, false
	}
	buf := list[len(list)-1]
	b.pool[key] = list[:len(list)-1]
	b.poolBytes -= buf.size
	b.stats.PoolHits++
	return buf, true
}

// poolPut offers a buffer to the pool, returning false if the pool is full.
func (b *Backend) poolPut(buf *buffer) bool {
	if b.noPool {
		return false
	}
	if b.poolBytes+buf.size > poolCap {
		return false
	}
	if b.pool == nil {
		b.pool = map[poolKey][]*buffer{}
	}
	key := poolKey{size: buf.size, usage: buf.usage, staging: buf.staging}
	b.pool[key] = append(b.pool[key], buf)
	b.poolBytes += buf.size
	return true
}

// Stats returns a snapshot of device activity counters.
func (b *Backend) Stats() Stats {
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.stats
	s.PooledBytes = b.poolBytes
	s.DeviceBytes = b.deviceBytes
	s.PeakDeviceBytes = b.peakBytes
	return s
}

// ResetStats clears the device activity counters. The peak byte gauge is reset
// to the current live size so subsequent measurements are relative to it.
func (b *Backend) ResetStats() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.stats = Stats{}
	b.peakBytes = b.deviceBytes
}

func (b *Backend) memoryType(bits, flags uint32) (uint32, bool) {
	for i := uint32(0); i < b.memProps.memoryTypeCount && i < 32; i++ {
		if bits&(1<<i) == 0 {
			continue
		}
		if b.memProps.memoryTypes[i].propertyFlags&flags == flags {
			return i, true
		}
	}
	return 0, false
}

// memoryTypeExclude finds a memory type matching flags but without exclude set.
func (b *Backend) memoryTypeExclude(bits, flags, exclude uint32) (uint32, bool) {
	for i := uint32(0); i < b.memProps.memoryTypeCount && i < 32; i++ {
		pf := b.memProps.memoryTypes[i].propertyFlags
		if bits&(1<<i) == 0 || pf&exclude != 0 || pf&flags != flags {
			continue
		}
		return i, true
	}
	return 0, false
}

// deviceMemorySize reports the total size of the device-local heaps, falling
// back to the largest heap when none is marked device-local. Zero means the
// size could not be determined.
func (b *Backend) deviceMemorySize() uint64 {
	return deviceLocalSize(b.memProps)
}

// pipeline lazily creates a compute pipeline from an embedded shader.
func (b *Backend) pipeline(name string) (uintptr, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if p, ok := b.pipelines[name]; ok {
		return p, nil
	}
	data, err := shaderFS.ReadFile("spirv/" + name + ".spv")
	if err != nil {
		return 0, fmt.Errorf("vulkan: shader %s: %w", name, err)
	}
	smci := shaderModuleCreateInfo{
		sType:    vkStructureShaderModuleCreateInfo,
		codeSize: uintptr(len(data)),
		pCode:    uintptr(unsafe.Pointer(&data[0])),
	}
	var module uintptr
	if res := vkCall(b.vk.CreateShaderModule, b.device, uintptr(unsafe.Pointer(&smci)), 0, uintptr(unsafe.Pointer(&module))); res != vkSuccess {
		return 0, fmt.Errorf("vulkan: create shader module %s failed", name)
	}
	entry, eb := cstr("main")
	stage := pipelineShaderStageCreateInfo{
		sType:  vkStructurePipelineShaderStageCi,
		stage:  vkShaderStageComputeBit,
		module: module,
		pName:  entry,
	}
	cpci := computePipelineCreateInfo{
		sType:  vkStructureComputePipelineCi,
		stage:  stage,
		layout: b.pipeLayout,
	}
	var pipe uintptr
	if res := vkCall(b.vk.CreateComputePipelines, b.device, 0, 1, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&pipe))); res != vkSuccess {
		runtime.KeepAlive(eb)
		return 0, fmt.Errorf("vulkan: create pipeline %s failed", name)
	}
	runtime.KeepAlive(eb)
	b.pipelines[name] = pipe
	return pipe, nil
}

// beginBatch starts recording compute commands if not already recording.
func (b *Backend) beginBatch() error {
	if b.recording {
		return nil
	}
	if res := vkCall(b.vk.ResetDescriptorPool, b.device, b.descPool, 0); res != vkSuccess {
		return fmt.Errorf("vulkan: reset descriptor pool failed")
	}
	if res := vkCall(b.vk.ResetCommandBuffer, b.cmdBuffer, 0); res != vkSuccess {
		return fmt.Errorf("vulkan: reset command buffer failed")
	}
	begin := commandBufferBeginInfo{sType: vkStructureCommandBufferBeginInfo, flags: vkCommandBufferUsageOneTime}
	if res := vkCall(b.vk.BeginCommandBuffer, b.cmdBuffer, uintptr(unsafe.Pointer(&begin))); res != vkSuccess {
		return fmt.Errorf("vulkan: begin command buffer failed")
	}
	b.recording = true
	return nil
}

// flush submits and waits for all recorded commands, then releases buffers that
// were freed during recording.
func (b *Backend) flush() error {
	if b.recording {
		if res := vkCall(b.vk.EndCommandBuffer, b.cmdBuffer); res != vkSuccess {
			return fmt.Errorf("vulkan: end command buffer failed")
		}
		submit := submitInfo{
			sType:              vkStructureSubmitInfo,
			commandBufferCount: 1,
			pCommandBuffers:    uintptr(unsafe.Pointer(&b.cmdBuffer)),
		}
		if res := vkCall(b.vk.ResetFences, b.device, 1, uintptr(unsafe.Pointer(&b.fence))); res != vkSuccess {
			return fmt.Errorf("vulkan: reset fence failed")
		}
		if res := vkCall(b.vk.QueueSubmit, b.queue, 1, uintptr(unsafe.Pointer(&submit)), b.fence); res != vkSuccess {
			return fmt.Errorf("vulkan: queue submit failed")
		}
		if res := vkCall(b.vk.WaitForFences, b.device, 1, uintptr(unsafe.Pointer(&b.fence)), 1, ^uintptr(0)); res != vkSuccess {
			return fmt.Errorf("vulkan: wait for fence failed")
		}
		b.stats.Submits++
		b.recording = false
	}
	for _, x := range b.pending {
		b.destroyBuffer(x)
	}
	b.pending = b.pending[:0]
	return nil
}

// dispatch records a compute dispatch into the current batch.
func (b *Backend) dispatch(name string, bindings []*buffer, push []byte, groups [3]uint32) error {
	b.stats.Dispatches++
	pipe, err := b.pipeline(name)
	if err != nil {
		return err
	}
	if err := b.beginBatch(); err != nil {
		return err
	}

	dsai := descriptorSetAllocateInfo{
		sType:              vkStructureDescriptorSetAi,
		descriptorPool:     b.descPool,
		descriptorSetCount: 1,
		pSetLayouts:        uintptr(unsafe.Pointer(&b.descLayout)),
	}
	var set uintptr
	if res := vkCall(b.vk.AllocateDescriptorSets, b.device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&set))); res != vkSuccess {
		return fmt.Errorf("vulkan: allocate descriptor set failed")
	}

	var infos [maxBindings]descriptorBufferInfo
	for i := 0; i < maxBindings; i++ {
		buf := b.dummy
		if i < len(bindings) && bindings[i] != nil {
			buf = bindings[i]
		}
		infos[i] = descriptorBufferInfo{buffer: buf.buf, offset: 0, rng: buf.size}
	}
	writes := make([]writeDescriptorSet, maxBindings)
	for i := 0; i < maxBindings; i++ {
		writes[i] = writeDescriptorSet{
			sType:           vkStructureWriteDescriptorSet,
			dstSet:          set,
			dstBinding:      uint32(i),
			descriptorCount: 1,
			descriptorType:  vkDescriptorTypeStorage,
			pBufferInfo:     uintptr(unsafe.Pointer(&infos[i])),
		}
	}
	vkCall(b.vk.UpdateDescriptorSets, b.device, uintptr(len(writes)), uintptr(unsafe.Pointer(&writes[0])), 0, 0)

	vkCall(b.vk.CmdBindPipeline, b.cmdBuffer, vkPipelineBindPointCompute, pipe)
	vkCall(b.vk.CmdBindDescriptorSets, b.cmdBuffer, vkPipelineBindPointCompute, b.pipeLayout, 0, 1, uintptr(unsafe.Pointer(&set)), 0, 0)
	if len(push) > 0 {
		vkCall(b.vk.CmdPushConstants, b.cmdBuffer, b.pipeLayout, vkShaderStageComputeBit, 0, uintptr(len(push)), uintptr(unsafe.Pointer(&push[0])))
	}
	vkCall(b.vk.CmdDispatch, b.cmdBuffer, uintptr(groups[0]), uintptr(groups[1]), uintptr(groups[2]))
	barrier := memoryBarrier{
		sType:         vkStructureMemoryBarrier,
		srcAccessMask: vkAccessShaderRead | vkAccessShaderWrite,
		dstAccessMask: vkAccessShaderRead | vkAccessShaderWrite,
	}
	vkCall(b.vk.CmdPipelineBarrier, b.cmdBuffer, vkPipelineStageCompute, vkPipelineStageCompute, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	return nil
}

// copyRows records a copy of a [rows,m] block into dst [outRows,m] starting at
// row rowOffset.
func (b *Backend) copyRows(dst, src *buffer, rows, m, outRows, rowOffset int) error {
	if err := b.beginBatch(); err != nil {
		return err
	}
	pipe, err := b.pipeline("copy_rows")
	if err != nil {
		return err
	}
	dsai := descriptorSetAllocateInfo{
		sType:              vkStructureDescriptorSetAi,
		descriptorPool:     b.descPool,
		descriptorSetCount: 1,
		pSetLayouts:        uintptr(unsafe.Pointer(&b.descLayout)),
	}
	var set uintptr
	if res := vkCall(b.vk.AllocateDescriptorSets, b.device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&set))); res != vkSuccess {
		return fmt.Errorf("vulkan: allocate descriptor set failed")
	}
	infos := [4]descriptorBufferInfo{
		{buffer: src.buf, rng: src.size},
		{buffer: dst.buf, rng: dst.size},
		{buffer: b.dummy.buf, rng: b.dummy.size},
		{buffer: b.dummy.buf, rng: b.dummy.size},
	}
	writes := make([]writeDescriptorSet, 4)
	for i := 0; i < 4; i++ {
		writes[i] = writeDescriptorSet{
			sType:           vkStructureWriteDescriptorSet,
			dstSet:          set,
			dstBinding:      uint32(i),
			descriptorCount: 1,
			descriptorType:  vkDescriptorTypeStorage,
			pBufferInfo:     uintptr(unsafe.Pointer(&infos[i])),
		}
	}
	vkCall(b.vk.UpdateDescriptorSets, b.device, uintptr(len(writes)), uintptr(unsafe.Pointer(&writes[0])), 0, 0)
	total := uint32(rows * m)
	vkCall(b.vk.CmdBindPipeline, b.cmdBuffer, vkPipelineBindPointCompute, pipe)
	vkCall(b.vk.CmdBindDescriptorSets, b.cmdBuffer, vkPipelineBindPointCompute, b.pipeLayout, 0, 1, uintptr(unsafe.Pointer(&set)), 0, 0)
	// Chunk the copy so the grid stays under maxComputeWorkGroupCount.
	const per = uint32(maxWorkGroups) * 64
	for base := uint32(0); base < total; base += per {
		n := total - base
		if n > per {
			n = per
		}
		groups := [3]uint32{ceilDiv(n, 64), 1, 1}
		p := push(uint32(rows), uint32(m), uint32(outRows), uint32(rowOffset), base)
		vkCall(b.vk.CmdPushConstants, b.cmdBuffer, b.pipeLayout, vkShaderStageComputeBit, 0, uintptr(len(p)), uintptr(unsafe.Pointer(&p[0])))
		vkCall(b.vk.CmdDispatch, b.cmdBuffer, uintptr(groups[0]), uintptr(groups[1]), uintptr(groups[2]))
	}
	barrier := memoryBarrier{
		sType:         vkStructureMemoryBarrier,
		srcAccessMask: vkAccessShaderRead | vkAccessShaderWrite,
		dstAccessMask: vkAccessShaderRead | vkAccessShaderWrite,
	}
	vkCall(b.vk.CmdPipelineBarrier, b.cmdBuffer, vkPipelineStageCompute, vkPipelineStageCompute, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	return nil
}

func push(puts ...any) []byte {
	buf := make([]byte, 128)
	off := 0
	for _, v := range puts {
		switch t := v.(type) {
		case uint32:
			binary.LittleEndian.PutUint32(buf[off:], t)
			off += 4
		case int:
			binary.LittleEndian.PutUint32(buf[off:], uint32(t))
			off += 4
		case float32:
			binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(t))
			off += 4
		}
	}
	return buf
}

// writeBytes copies src into a buffer, through a host-visible staging buffer
// and a recorded copy when the destination is device-local and not mapped. The
// staging buffer is released on the next flush.
func (b *Backend) writeBytes(x *buffer, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	b.stats.BytesUploaded += uint64(len(src))
	if x.mapped != nil {
		copy(unsafe.Slice((*byte)(x.mapped), len(src)), src)
		return nil
	}
	staging, err := b.allocHostBuffer(uint64(len(src)), vkBufferUsageTransferSrc|vkBufferUsageTransferDst)
	if err != nil {
		return err
	}
	copy(unsafe.Slice((*byte)(staging.mapped), len(src)), src)
	if err := b.beginBatch(); err != nil {
		b.destroyBuffer(staging)
		return err
	}
	region := bufferCopy{srcOffset: 0, dstOffset: 0, size: uint64(len(src))}
	vkCall(b.vk.CmdCopyBuffer, b.cmdBuffer, staging.buf, x.buf, 1, uintptr(unsafe.Pointer(&region)))
	// Make the transfer visible to the compute stage before any dispatch reads
	// the buffer.
	barrier := memoryBarrier{
		sType:         vkStructureMemoryBarrier,
		srcAccessMask: vkAccessTransferWrite,
		dstAccessMask: vkAccessShaderRead | vkAccessShaderWrite,
	}
	vkCall(b.vk.CmdPipelineBarrier, b.cmdBuffer, vkPipelineStageTransfer, vkPipelineStageCompute, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	b.Free(staging) // deferred until the flush submits the copy
	return nil
}

// readBytes returns a host copy of the buffer's contents, through a staging
// buffer when the source is not host-visible. It flushes pending commands.
func (b *Backend) readBytes(x *buffer, n int) ([]byte, error) {
	b.stats.BytesDownloaded += uint64(n)
	if x.mapped != nil {
		out := make([]byte, n)
		copy(out, unsafe.Slice((*byte)(x.mapped), n))
		return out, nil
	}
	if err := b.beginBatch(); err != nil {
		return nil, err
	}
	staging, err := b.allocHostBuffer(uint64(n), vkBufferUsageTransferDst|vkBufferUsageTransferSrc)
	if err != nil {
		return nil, err
	}
	region := bufferCopy{srcOffset: 0, dstOffset: 0, size: uint64(n)}
	// Make prior compute writes visible to the transfer stage.
	barrier := memoryBarrier{
		sType:         vkStructureMemoryBarrier,
		srcAccessMask: vkAccessShaderWrite,
		dstAccessMask: vkAccessTransferRead,
	}
	vkCall(b.vk.CmdPipelineBarrier, b.cmdBuffer, vkPipelineStageCompute, vkPipelineStageTransfer, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	vkCall(b.vk.CmdCopyBuffer, b.cmdBuffer, x.buf, staging.buf, 1, uintptr(unsafe.Pointer(&region)))
	if err := b.flush(); err != nil {
		b.destroyBuffer(staging)
		return nil, err
	}
	out := make([]byte, n)
	copy(out, unsafe.Slice((*byte)(staging.mapped), n))
	b.destroyBuffer(staging)
	return out, nil
}

func (b *Backend) uploadTensor(t *compute.Tensor) (*buffer, error) {
	buf, err := b.allocBuffer(uint64(len(t.F32))*4, vkBufferUsageStorage|vkBufferUsageTransferSrc|vkBufferUsageTransferDst)
	if err != nil {
		return nil, err
	}
	buf.dims = append([]int(nil), t.Dims...)
	buf.typ = t.Type
	if err := b.writeBytes(buf, unsafe.Slice((*byte)(unsafe.Pointer(&t.F32[0])), len(t.F32)*4)); err != nil {
		return nil, err
	}
	return buf, nil
}

func (b *Backend) downloadBuffer(x *buffer) (*compute.Tensor, error) {
	n := x.NumElements()
	raw, err := b.readBytes(x, n*4)
	if err != nil {
		return nil, err
	}
	out := &compute.Tensor{Dims: append([]int(nil), x.dims...), Type: x.typ, F32: make([]float32, n)}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out.F32[0])), n*4), raw)
	return out, nil
}

// vkCall invokes a Vulkan entry point. The uintptrescapes directive makes the
// compiler keep any pointer passed as uintptr alive and off the moving stack
// for the duration of the call; without it a goroutine stack growth during the
// cgo call corrupts the argument/result structs passed to the driver.
//
//go:uintptrescapes
func vkCall(fn uintptr, args ...uintptr) uintptr {
	r, _, _ := purego.SyscallN(fn, args...)
	return r
}
