TARGET = main 
SRC    = main.cu 

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)