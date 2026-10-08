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
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

//go:embed spirv/*.spv
var shaderFS embed.FS

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
	vkDescriptorPoolFreeSet      = 0x00000001
	vkApiVersion10               = 0x00400000
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

	CreateInstance                    uintptr
	EnumeratePhysicalDevices          uintptr
	GetPhysicalDeviceQueueFamilyProps uintptr
	GetPhysicalDeviceMemoryProps      uintptr
	GetPhysicalDeviceProperties       uintptr
	CreateDevice                      uintptr
	GetDeviceQueue                    uintptr
	CreateBuffer                      uintptr
	GetBufferMemoryRequirements       uintptr
	AllocateMemory                    uintptr
	BindBufferMemory                  uintptr
	MapMemory                         uintptr
	UnmapMemory                       uintptr
	DestroyBuffer                     uintptr
	FreeMemory                        uintptr
	CreateShaderModule                uintptr
	CreateDescriptorSetLayout         uintptr
	CreatePipelineLayout              uintptr
	CreateComputePipelines            uintptr
	CreateDescriptorPool              uintptr
	AllocateDescriptorSets            uintptr
	ResetDescriptorPool               uintptr
	UpdateDescriptorSets              uintptr
	CreateCommandPool                 uintptr
	AllocateCommandBuffers            uintptr
	ResetCommandBuffer                uintptr
	BeginCommandBuffer                uintptr
	CmdBindPipeline                   uintptr
	CmdBindDescriptorSets             uintptr
	CmdPushConstants                  uintptr
	CmdDispatch                       uintptr
	CmdPipelineBarrier                uintptr
	CmdCopyBuffer                     uintptr
	EndCommandBuffer                  uintptr
	QueueSubmit                       uintptr
	QueueWaitIdle                     uintptr
	CreateFence                       uintptr
	WaitForFences                     uintptr
	ResetFences                       uintptr
	DeviceWaitIdle                    uintptr
	DestroyDevice                     uintptr
	DestroyInstance                   uintptr
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
		&v.CreateInstance:                    "vkCreateInstance",
		&v.EnumeratePhysicalDevices:          "vkEnumeratePhysicalDevices",
		&v.GetPhysicalDeviceQueueFamilyProps: "vkGetPhysicalDeviceQueueFamilyProperties",
		&v.GetPhysicalDeviceMemoryProps:      "vkGetPhysicalDeviceMemoryProperties",
		&v.GetPhysicalDeviceProperties:       "vkGetPhysicalDeviceProperties",
		&v.CreateDevice:                      "vkCreateDevice",
		&v.GetDeviceQueue:                    "vkGetDeviceQueue",
		&v.CreateBuffer:                      "vkCreateBuffer",
		&v.GetBufferMemoryRequirements:       "vkGetBufferMemoryRequirements",
		&v.AllocateMemory:                    "vkAllocateMemory",
		&v.BindBufferMemory:                  "vkBindBufferMemory",
		&v.MapMemory:                         "vkMapMemory",
		&v.UnmapMemory:                       "vkUnmapMemory",
		&v.DestroyBuffer:                     "vkDestroyBuffer",
		&v.FreeMemory:                        "vkFreeMemory",
		&v.CreateShaderModule:                "vkCreateShaderModule",
		&v.CreateDescriptorSetLayout:         "vkCreateDescriptorSetLayout",
		&v.CreatePipelineLayout:              "vkCreatePipelineLayout",
		&v.CreateComputePipelines:            "vkCreateComputePipelines",
		&v.CreateDescriptorPool:              "vkCreateDescriptorPool",
		&v.AllocateDescriptorSets:            "vkAllocateDescriptorSets",
		&v.ResetDescriptorPool:               "vkResetDescriptorPool",
		&v.UpdateDescriptorSets:              "vkUpdateDescriptorSets",
		&v.CreateCommandPool:                 "vkCreateCommandPool",
		&v.AllocateCommandBuffers:            "vkAllocateCommandBuffers",
		&v.ResetCommandBuffer:                "vkResetCommandBuffer",
		&v.BeginCommandBuffer:                "vkBeginCommandBuffer",
		&v.CmdBindPipeline:                   "vkCmdBindPipeline",
		&v.CmdBindDescriptorSets:             "vkCmdBindDescriptorSets",
		&v.CmdPushConstants:                  "vkCmdPushConstants",
		&v.CmdDispatch:                       "vkCmdDispatch",
		&v.CmdPipelineBarrier:                "vkCmdPipelineBarrier",
		&v.CmdCopyBuffer:                     "vkCmdCopyBuffer",
		&v.EndCommandBuffer:                  "vkEndCommandBuffer",
		&v.QueueSubmit:                       "vkQueueSubmit",
		&v.QueueWaitIdle:                     "vkQueueWaitIdle",
		&v.CreateFence:                       "vkCreateFence",
		&v.WaitForFences:                     "vkWaitForFences",
		&v.ResetFences:                       "vkResetFences",
		&v.DeviceWaitIdle:                    "vkDeviceWaitIdle",
		&v.DestroyDevice:                     "vkDestroyDevice",
		&v.DestroyInstance:                   "vkDestroyInstance",
	}
	for ptr, n := range names {
		*ptr = v.load(instance, n)
	}
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

	mu        sync.Mutex
	pipelines map[string]uintptr
	closed    bool

	// maxScratchFloats bounds the float32 dequant scratch. It is overridable
	// in tests to exercise the row-blocking path.
	maxScratchFloats int
}

type buffer struct {
	b      *Backend
	buf    uintptr
	mem    uintptr
	mapped unsafe.Pointer
	size   uint64
	dims   []int
	typ    quant.Type
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
		apiVersion:         vkApiVersion10,
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

	selected := false
	for _, pd := range devs {
		if qf, ok := pickComputeQueue(v, pd); ok {
			b.physicalDevice = pd
			b.queueFamily = qf
			selected = true
			break
		}
	}
	if !selected {
		b.Close()
		return nil, errors.New("vulkan: no device with a compute queue")
	}

	// Device name for diagnostics.
	var props physicalDeviceProperties
	vkCall(v.GetPhysicalDeviceProperties, b.physicalDevice, uintptr(unsafe.Pointer(&props)))
	name := string(props.deviceName[:])
	for i, c := range name {
		if c == 0 {
			name = name[:i]
			break
		}
	}

	vkCall(v.GetPhysicalDeviceMemoryProps, b.physicalDevice, uintptr(unsafe.Pointer(&b.memProps)))

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
	var device uintptr
	if res := vkCall(v.CreateDevice, b.physicalDevice, uintptr(unsafe.Pointer(&dci)), 0, uintptr(unsafe.Pointer(&device))); res != vkSuccess {
		b.Close()
		return nil, fmt.Errorf("vulkan: vkCreateDevice failed (%d)", int32(res))
	}
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
		Name:          "vulkan(" + name + ")",
		MaxBufferSize: math.MaxUint32,
	}

	d, err := b.allocBuffer(16, vkBufferUsageStorage|vkBufferUsageTransferSrc|vkBufferUsageTransferDst)
	if err != nil {
		b.Close()
		return nil, err
	}
	b.dummy = d
	return b, nil
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
	bindings := [4]descriptorSetLayoutBinding{}
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

	poolSize := descriptorPoolSize{typ: vkDescriptorTypeStorage, descriptorCount: 4}
	dpci := descriptorPoolCreateInfo{
		sType:         vkStructureDescriptorPoolCi,
		flags:         vkDescriptorPoolFreeSet,
		maxSets:       1,
		poolSizeCount: 1,
		pPoolSizes:    uintptr(unsafe.Pointer(&poolSize)),
	}
	if res := vkCall(b.vk.CreateDescriptorPool, b.device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&b.descPool))); res != vkSuccess {
		return fmt.Errorf("vulkan: create descriptor pool failed")
	}
	return nil
}

func (b *Backend) allocBuffer(size uint64, usage uint32) (*buffer, error) {
	if size == 0 {
		size = 4
	}
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

	idx, ok := b.memoryType(req.memoryTypeBits, vkMemoryPropertyHostVisible|vkMemoryPropertyHostCoherent)
	if !ok {
		vkCall(b.vk.DestroyBuffer, b.device, buf, 0)
		return nil, errors.New("vulkan: no host visible memory type")
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
	var mapped unsafe.Pointer
	if res := vkCall(b.vk.MapMemory, b.device, mem, 0, uintptr(size), 0, uintptr(unsafe.Pointer(&mapped))); res != vkSuccess {
		vkCall(b.vk.FreeMemory, b.device, mem, 0)
		vkCall(b.vk.DestroyBuffer, b.device, buf, 0)
		return nil, fmt.Errorf("vulkan: map memory failed")
	}
	return &buffer{b: b, buf: buf, mem: mem, mapped: mapped, size: size, typ: quant.TypeF32}, nil
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

// dispatch runs a compute pipeline with the given bindings and push constants.
func (b *Backend) dispatch(name string, bindings []*buffer, push []byte, groups [3]uint32) error {
	pipe, err := b.pipeline(name)
	if err != nil {
		return err
	}

	if res := vkCall(b.vk.ResetDescriptorPool, b.device, b.descPool, 0); res != vkSuccess {
		return fmt.Errorf("vulkan: reset descriptor pool failed")
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

	var infos [4]descriptorBufferInfo
	for i := 0; i < 4; i++ {
		buf := b.dummy
		if i < len(bindings) && bindings[i] != nil {
			buf = bindings[i]
		}
		infos[i] = descriptorBufferInfo{buffer: buf.buf, offset: 0, rng: buf.size}
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

	// Record.
	if res := vkCall(b.vk.ResetCommandBuffer, b.cmdBuffer, 0); res != vkSuccess {
		return fmt.Errorf("vulkan: reset command buffer failed")
	}
	begin := commandBufferBeginInfo{sType: vkStructureCommandBufferBeginInfo, flags: vkCommandBufferUsageOneTime}
	if res := vkCall(b.vk.BeginCommandBuffer, b.cmdBuffer, uintptr(unsafe.Pointer(&begin))); res != vkSuccess {
		return fmt.Errorf("vulkan: begin command buffer failed")
	}
	vkCall(b.vk.CmdBindPipeline, b.cmdBuffer, vkPipelineBindPointCompute, pipe)
	vkCall(b.vk.CmdBindDescriptorSets, b.cmdBuffer, vkPipelineBindPointCompute, b.pipeLayout, 0, 1, uintptr(unsafe.Pointer(&set)), 0, 0)
	if len(push) > 0 {
		vkCall(b.vk.CmdPushConstants, b.cmdBuffer, b.pipeLayout, vkShaderStageComputeBit, 0, uintptr(len(push)), uintptr(unsafe.Pointer(&push[0])))
	}
	vkCall(b.vk.CmdDispatch, b.cmdBuffer, uintptr(groups[0]), uintptr(groups[1]), uintptr(groups[2]))
	barrier := memoryBarrier{
		sType:         vkStructureMemoryBarrier,
		srcAccessMask: vkAccessShaderWrite,
		dstAccessMask: vkAccessShaderRead | vkAccessShaderWrite,
	}
	vkCall(b.vk.CmdPipelineBarrier, b.cmdBuffer, vkPipelineStageCompute, vkPipelineStageCompute, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
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

func (b *Backend) uploadTensor(t *compute.Tensor) (*buffer, error) {
	buf, err := b.allocBuffer(uint64(len(t.F32))*4, vkBufferUsageStorage|vkBufferUsageTransferSrc|vkBufferUsageTransferDst)
	if err != nil {
		return nil, err
	}
	buf.dims = append([]int(nil), t.Dims...)
	buf.typ = t.Type
	src := unsafe.Slice((*byte)(unsafe.Pointer(&t.F32[0])), len(t.F32)*4)
	dst := unsafe.Slice((*byte)(buf.mapped), int(buf.size))
	copy(dst, src)
	return buf, nil
}

func (b *Backend) downloadBuffer(x *buffer) *compute.Tensor {
	n := x.NumElements()
	out := &compute.Tensor{Dims: append([]int(nil), x.dims...), Type: x.typ, F32: make([]float32, n)}
	src := unsafe.Slice((*byte)(x.mapped), n*4)
	dst := unsafe.Slice((*byte)(unsafe.Pointer(&out.F32[0])), n*4)
	copy(dst, src)
	return out
}

func vkCall(fn uintptr, args ...uintptr) uintptr {
	r, _, _ := purego.SyscallN(fn, args...)
	return r
}
