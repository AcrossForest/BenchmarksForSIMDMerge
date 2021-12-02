#include <stdint.h>

void m5_arm(uint64_t address){return;}
void m5_quiesce(void){return;}
void m5_quiesce_ns(uint64_t ns){return;}
void m5_quiesce_cycle(uint64_t cycles){return;}
uint64_t m5_quiesce_time(void){return 0;}
uint64_t m5_rpns(){return 0;}
void m5_wake_cpu(uint64_t cpuid){return;}

void m5_exit(uint64_t ns_delay){return;}
void m5_fail(uint64_t ns_delay, uint64_t code){return;}
// m5_sum is for sanity checking the gem5 op interface.
unsigned m5_sum(unsigned a, unsigned b, unsigned c,
                unsigned d, unsigned e, unsigned f){return 0;}
uint64_t m5_init_param(uint64_t key_str1, uint64_t key_str2){return 0;}
void m5_checkpoint(uint64_t ns_delay, uint64_t ns_period){return;}
void m5_reset_stats(uint64_t ns_delay, uint64_t ns_period){return;}
void m5_dump_stats(uint64_t ns_delay, uint64_t ns_period){return;}
void m5_dump_reset_stats(uint64_t ns_delay, uint64_t ns_period){return;}
uint64_t m5_read_file(void *buffer, uint64_t len, uint64_t offset){return 0;}
uint64_t m5_write_file(void *buffer, uint64_t len, uint64_t offset,
                       const char *filename){return 0;}
void m5_debug_break(void){return;}
void m5_switch_cpu(void){return;}
void m5_dist_toggle_sync(void){return;}
void m5_add_symbol(uint64_t addr, const char *symbol){return;}
void m5_load_symbol(){return;}
void m5_panic(void){return;}
void m5_work_begin(uint64_t workid, uint64_t threadid){return;}
void m5_work_end(uint64_t workid, uint64_t threadid){return;}

void m5_se_syscall(){return;}
void m5_se_page_fault(){return;}