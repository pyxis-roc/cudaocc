CUDA_INSTALL_PATH=/usr/local/cuda
BUILD=.

all: $(BUILD)/cudaocc_helper.so

$(BUILD)/cudaocc_helper.so: $(BUILD)/cudaocc_helper.o
	$(CC) -shared -o $@ $<

$(BUILD)/cudaocc_helper.o: cudaocc_helper.c
	$(CC) -I$(CUDA_INSTALL_PATH)/include -fPIC $< -c -o $@
