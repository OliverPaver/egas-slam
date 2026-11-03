#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

// Link against exactly the trainer's transitive libraries. No GPU is needed.
int main()
{
    namespace fs = std::filesystem;
    auto *address = reinterpret_cast<void *>(
        static_cast<std::uintmax_t (*)(const fs::path &, std::error_code &)>(fs::remove_all));
    Dl_info origin{};
    if (!dladdr(address, &origin) || !origin.dli_fname ||
        std::string(origin.dli_fname).find("libstdc++") == std::string::npos)
    {
        std::cerr << "Filesystem removal did not resolve to the host C++ runtime\n";
        return 1;
    }
    std::cout << "Filesystem provider: " << origin.dli_fname << std::endl;
    for (bool use_error_code : {true, false})
    {
        char pattern[] = "/tmp/egas-filesystem-XXXXXX";
        const char *created = mkdtemp(pattern);
        if (!created)
            throw std::runtime_error("Could not create disposable filesystem fixture");
        const fs::path root(created);
        fs::create_directories(root / "submap" / "nested");
        std::ofstream(root / "submap" / "nested" / "payload.txt") << "test payload";
        std::error_code error;
        const auto removed = use_error_code ? fs::remove_all(root, error) : fs::remove_all(root);
        if (error || removed != 4 || fs::exists(root))
        {
            std::cerr << "Nested filesystem removal failed: " << error.message() << '\n';
            return 1;
        }
    }
    std::cout << "PASS: both recursive-removal overloads\n";
    return 0;
}
