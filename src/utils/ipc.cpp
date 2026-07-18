/**************************************************************
 * IPC helpers: RAII wrappers for System V shared memory and semaphores.
 **************************************************************/

#include "ipc.h"

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>

namespace utils {

// ===================== Semaphore =====================

int sem_op(int semid, int op) {
  struct sembuf buf{};
  buf.sem_num = 0;
  buf.sem_op = op;
  buf.sem_flg = 0;
  return semop(semid, &buf, 1);
}

Semaphore::Semaphore(key_t key, int init_val) {
  semid_ = semget(key, 1, IPC_CREAT | 0666);
  if (semid_ < 0) {
    perror("semget failed");
    exit(EXIT_FAILURE);
  }
  if (semctl(semid_, 0, SETVAL, init_val) < 0) {
    perror("semctl SETVAL failed");
    exit(EXIT_FAILURE);
  }
}

Semaphore::~Semaphore() { semctl(semid_, 0, IPC_RMID); }

// ===================== Shared memory =====================

void* attach_shm(int shmid) {
  void* addr = shmat(shmid, nullptr, 0);
  if (addr == (void*)-1) {
    perror("shmat failed");
    exit(EXIT_FAILURE);
  }
  return addr;
}

void detach_shm(void* ptr) { shmdt(ptr); }

SharedMem::SharedMem(key_t key, size_t size) {
  shmid_ = shmget(key, size, IPC_CREAT | 0666);
  if (shmid_ < 0 && errno == EINVAL) {
    // A stale segment with the same key but a smaller size exists (e.g. left
    // over from a crashed run with a different ShmBlock layout). Remove it and
    // recreate at the requested size.
    int old = shmget(key, 0, 0666);
    if (old >= 0) {
      shmctl(old, IPC_RMID, nullptr);
    }
    shmid_ = shmget(key, size, IPC_CREAT | 0666);
  }
  if (shmid_ < 0) {
    perror("shmget failed");
    exit(EXIT_FAILURE);
  }
  ptr_ = attach_shm(shmid_);
}

SharedMem::~SharedMem() {
  shmdt(ptr_);
  shmctl(shmid_, IPC_RMID, nullptr);
}

} // namespace utils
