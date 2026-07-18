/**************************************************************
 * IPC helpers: RAII wrappers for System V shared memory and semaphores.
 **************************************************************/

#pragma once

#include <cstddef>
#include <sys/types.h>

namespace utils {

// ===================== Semaphore =====================

// Perform a single semaphore operation (op < 0 to acquire, op > 0 to release).
int sem_op(int semid, int op);

// RAII critical-section guard: P(-1) on construction, V(+1) on destruction.
struct SemGuard {
  int semid;
  explicit SemGuard(int s) : semid(s) { sem_op(semid, -1); }
  ~SemGuard() { sem_op(semid, 1); }
  SemGuard(const SemGuard&) = delete;
  SemGuard& operator=(const SemGuard&) = delete;
};

// RAII semaphore: create and initialize on construction, IPC_RMID on destruction.
class Semaphore {
 public:
  explicit Semaphore(key_t key, int init_val = 1);
  ~Semaphore();
  Semaphore(const Semaphore&) = delete;
  Semaphore& operator=(const Semaphore&) = delete;

  int id() const { return semid_; }

 private:
  int semid_;
};

// ===================== Shared memory =====================

// Attach to an existing shared-memory segment by id. Aborts on failure.
// Exposed for child processes that only have the shmid (e.g. after fork).
void* attach_shm(int shmid);

// Detach a shared-memory segment previously returned by attach_shm().
// Does not remove the segment (IPC_RMID); the owner's SharedMem destructor does that.
void detach_shm(void* ptr);

// RAII shared memory: create and attach on construction, detach + IPC_RMID on destruction.
// Manages a raw byte block; callers cast get() to their own struct type.
class SharedMem {
 public:
  SharedMem(key_t key, size_t size);
  ~SharedMem();
  SharedMem(const SharedMem&) = delete;
  SharedMem& operator=(const SharedMem&) = delete;

  int id() const { return shmid_; }
  void* get() const { return ptr_; }

 private:
  int shmid_;
  void* ptr_;
};

} // namespace utils
