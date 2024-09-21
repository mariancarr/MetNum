CXX = g++
CXXFLAGS = -std=c++11 -Wall -g -O3

BUILD_DIR = ./build

TARGET = main

all: build $(BUILD_DIR)/$(TARGET)
	$(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LDFLAGS)

$(BUILD_DIR)/%: %.cpp $(OBJECTS:%=$(BUILD_DIR)/%)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build:
	@mkdir -p $(BUILD_DIR)



debug: CXXFLAGS += -DDEBUG -g
debug: all

clean:
	-@rm -rvf $(BUILD_DIR)/*
