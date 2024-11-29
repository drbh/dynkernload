CXX = g++
CXXFLAGS = -O3 -fPIC -march=native

TARGET = libmatmul.so

$(TARGET): matmul.cpp
	$(CXX) $(CXXFLAGS) -shared $< -o $@

clean:
	rm -f $(TARGET)