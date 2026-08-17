CXX ?= c++
CXXFLAGS ?= -O3
CXXFLAGS += -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS ?=

TARGET := validate_routes
SOURCES := main.cpp artifacts.cpp checks.cpp
OBJECTS := $(SOURCES:.cpp=.o)

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

%.o: %.cpp validation.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(TARGET)
	python3 tests/run_tests.py --validator ./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)
