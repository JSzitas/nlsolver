all: debug example

debug:
	clang++ -std=c++17 -O0 -g example.cpp -o debug
example:
	clang++ -std=c++17 -O3 -Wall -march=native example.cpp -o example
clean:
	rm -rf debug example


