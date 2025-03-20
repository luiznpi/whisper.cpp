To build this do: 

//SDL2 needs to be installed in the mac at least. Maybe also PC

From root folder clear the build folder by 
rm -rf build
mkdir build && cd build

Create build files with 
cmake -DBUILD_SHARED_LIBS=ON ..

Then go to root using 
cd ..
Then make library using 
make