#!/bin/bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0 ./build/viz "$@"