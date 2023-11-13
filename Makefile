all: debug example debug_simd

debug:
	clang++ -std=c++17 -O0 -g -Wall example.cpp -o debug
example:
	clang++ -std=c++17 -O3 -Wall -march=native example.cpp -o example
debug_simd:
	clang++ -std=c++17 -O0 -g -Wall -march=native simd_debug.cpp -o simd_debug
example_simd:
	clang++ -std=c++17 -O0 -g -Wall -march=native example.cpp -o example_simd
clean:
	rm -rf debug example debug_simd


