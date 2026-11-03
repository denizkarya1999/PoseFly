/* Small GNU/Linux executable for file managers that do not run .desktop files
 * inside project folders. Resolve the project relative to this executable.
 * Rebuild from the project root:
 *   cc -O2 -Wall -Wextra -Werror linux/launcher.c -o PoseFly
 */
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    char executable[PATH_MAX];
    char script[PATH_MAX];
    ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length < 0 || length >= (ssize_t)sizeof(executable) - 1) {
        fprintf(stderr, "PoseFly: cannot resolve launcher location.\n");
        return EXIT_FAILURE;
    }
    executable[length] = '\0';
    char *slash = strrchr(executable, '/');
    if (slash == NULL)
        return EXIT_FAILURE;
    *slash = '\0';
    int count = snprintf(script, sizeof(script),
                         "%s/PoseFly Desktop App/PoseFly.WinForms/PyBackend/RUN_POSEFLY_LINUX.sh",
                         executable);
    if (count < 0 || (size_t)count >= sizeof(script)) {
        fprintf(stderr, "PoseFly: project path is too long.\n");
        return EXIT_FAILURE;
    }
    char **args = calloc((size_t)argc + 2, sizeof(*args));
    if (args == NULL) {
        perror("PoseFly");
        return EXIT_FAILURE;
    }
    args[0] = "bash";
    args[1] = script;
    for (int i = 1; i < argc; ++i)
        args[i + 1] = argv[i];
    execv("/bin/bash", args);
    perror("PoseFly: unable to start launcher");
    free(args);
    return EXIT_FAILURE;
}
