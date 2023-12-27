all:
	g++ -I include -L lib -o bin\main src\main.cxx -lmingw32 -lSDL2main -lSDL2 -lSDL2_image