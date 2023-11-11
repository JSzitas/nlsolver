all: debug example

debug:
	clang++ -std=c++17 -O0 -g example.cpp -o debug
example:
	clang++ -std=c++17 -O3 -Wall -march=native example.cpp -o example
debug_simd:
	clang++ -std=c++17 -O0 -g -march=native simd_debug.cpp -o simd_debug
clean:
	rm -rf debug example


