all: lib/cudaocc_helper.so bin/gen_device_prop

lib/cudaocc_helper.so:
	make -C src/cudaocc_helper BUILD=../../lib/

bin/gen_device_prop:
	make -C src/gen_device_prop BUILD=../../bin

.PHONY: install

install: bin/gen_device_prop lib/cudaocc_helper.so
	cp -v bin/gen_device_prop $(VIRTUAL_ENV)/bin
	@echo export LD_LIBRARY_PATH=`pwd`/lib:\$$LD_LIBRARY_PATH
