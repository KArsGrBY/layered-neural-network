#ifndef TOOLS_H
#define TOOLS_H

template <typename T>
T read(std::ifstream & fin) {
    T result;
    fin.read(reinterpret_cast<char*>(&result), sizeof(T));
    return result;
}

template <typename T>
void write(std::ofstream & fout, T value) {
    fout.write(reinterpret_cast<char*>(&value), sizeof(value));
}

#endif //TOOLS_H
