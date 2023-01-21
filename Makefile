CPP = clang++
CPPFLAGS  = -std=c++17 -O3 -Wall -march=native

TARGET = example

all: $(TARGET)

$(TARGET): $(TARGET).cpp
		$(CPP) $(CPPFLAGS) -o $(TARGET) $(TARGET).cpp

clean:
	$(RM) $(TARGET)