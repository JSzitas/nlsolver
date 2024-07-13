all: debug example example_simd

debug:
	clang++ -std=c++17 -O0 -ggdb -Wall example.cpp -o debug
example:
	clang++ -std=c++17 -O3 -Wall -march=native example.cpp -o example
example_simd:
	clang++ -std=c++17 -O0 -g -Wall -march=native example.cpp -o example_simd
test:
	clang++ -std=c++17 -O2 -g -Wall tests.cpp -o tests; ./tests; rm tests
roots:
	clang++ -std=c++17 -O2 -g -Wall roots.cpp -o roots; ./roots; rm roots
clean:
	rm -rf debug example example_simd


