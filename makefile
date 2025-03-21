clean:
	cargo clean

build:
	maturin develop

test:
	pytest py_test/

all: 
clean build test