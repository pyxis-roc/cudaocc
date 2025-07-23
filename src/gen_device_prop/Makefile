NVCC:=nvcc
BUILD:=.

all: $(BUILD)/gen_device_prop

$(BUILD)/gen_device_prop: gen_device_prop.cu
	$(NVCC) $< -o $@
