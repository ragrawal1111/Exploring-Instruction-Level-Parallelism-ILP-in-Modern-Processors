# Copyright (c) 2012-2013 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Copyright (c) 2006-2008 The Regents of The University of Michigan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Simple test script
#
# "m5 test3.py" - Customized for O3CPU pipeline parameters.

import argparse
import os
import sys

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.params import NULL
from m5.util import (
    addToPath,
    fatal,
    warn,
)

from gem5.isas import ISA

addToPath("../../")

from common import (
    CacheConfig,
    CpuConfig,
    MemConfig,
    ObjectList,
    Options,
    Simulation,
)
from common.Caches import *
from common.cpu2000 import *
from common.FileSystemConfig import config_filesystem
from ruby import Ruby


def get_processes(args):
    """Interprets provided args and returns a list of processes"""

    multiprocesses = []
    inputs = []
    outputs = []
    errouts = []
    pargs = []

    workloads = args.cmd.split(";")
    if args.input != "":
        inputs = args.input.split(";")
    if args.output != "":
        outputs = args.output.split(";")
    if args.errout != "":
        errouts = args.errouts.split(";")
    if args.options != "":
        pargs = args.options.split(";")

    idx = 0
    for wrkld in workloads:
        process = Process(pid=100 + idx)
        process.executable = wrkld
        process.cwd = os.getcwd()
        process.gid = os.getgid()

        if args.env:
            with open(args.env) as f:
                process.env = [line.rstrip() for line in f]

        if len(pargs) > idx:
            process.cmd = [wrkld] + pargs[idx].split()
        else:
            process.cmd = [wrkld]

        if len(inputs) > idx:
            process.input = inputs[idx]
        if len(outputs) > idx:
            process.output = outputs[idx]
        if len(errouts) > idx:
            process.errout = errouts[idx]

        multiprocesses.append(process)
        idx += 1

    if args.smt:
        cpu_type = ObjectList.cpu_list.get(args.cpu_type)
        assert ObjectList.is_o3_cpu(cpu_type), "SMT requires an O3CPU"
        return multiprocesses, idx
    else:
        return multiprocesses, 1


warn(
    "The se.py script is deprecated. It will be removed in future releases of "
    " gem5."
)

parser = argparse.ArgumentParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)

# Added CPU pipeline specific options for O3CPU
parser.add_argument(
    "--issue-width",
    type=int,
    default=4,
    help="Issue width for O3 CPUs (e.g., O3CPU)",
)
parser.add_argument(
    "--decode-width",
    type=int,
    default=4,
    help="Decode width for O3 CPUs (e.g., O3CPU)",
)
parser.add_argument(
    "--rename-width",
    type=int,
    default=4,
    help="Rename width for O3 CPUs (e.g., O3CPU)",
)
parser.add_argument(
    "--commit-width",
    type=int,
    default=4,
    help="Commit width for O3 CPUs (e.g., O3CPU)",
)
parser.add_argument(
    "--fetch-width",
    type=int,
    default=4,
    help="Fetch width for O3 CPUs (e.g., O3CPU and X86MinorCPU)",
)
parser.add_argument(
    "--num-int-alu",
    type=int,
    default=4,
    help="Number of integer ALUs for O3 CPUs",
)
parser.add_argument(
    "--num-fp-alu",
    type=int,
    default=1,
    help="Number of floating point ALUs for O3 CPUs",
)
parser.add_argument(
    "--num-iq-entries",
    type=int,
    default=64,
    help="Number of instruction queue entries for O3 CPUs",
)
parser.add_argument(
    "--num-rob-entries",
    type=int,
    default=192,
    help="Number of reorder buffer entries for O3 CPUs",
)
parser.add_argument(
    "--num-lsq-entries",
    type=int,
    default=64,
    help="Number of load-store queue entries for O3 CPUs",
)


if "--ruby" in sys.argv:
    Ruby.define_options(parser)

args = parser.parse_args()

multiprocesses = []
numThreads = 1

if args.bench:
    apps = args.bench.split("-")
    if len(apps) != args.num_cpus:
        print("number of benchmarks not equal to set num_cpus!")
        sys.exit(1)

    for app in apps:
        try:
            if buildEnv["TARGET_ISA"] == "arm":
                 exec(
                     "workload = %s('arm_%s', 'linux', '%s')"
                     % (app, args.arm_iset, args.spec_input)
                 )
            else:
                exec(
                    "workload = %s('%s', 'linux', '%s')"
                    % (app, buildEnv['TARGET_ISA'].lower(), args.spec_input)
                )
            multiprocesses.append(workload.makeProcess())
        except Exception as e:
            print(
                f"Unable to find workload for ISA: {app}. Error: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
elif args.cmd:
    multiprocesses, numThreads = get_processes(args)
else:
    print("No workload specified. Exiting!\n", file=sys.stderr)
    sys.exit(1)


(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(args)

# *** IMPORTANT: Set O3CPU parameters directly on CPUClass BEFORE instantiation ***
# If the parameters are truly directly on X86O3CPU as Params, this is the most
# direct way to set them BEFORE the C++ object is fully constructed.
# We'll put try-except blocks to catch specific failures.

if ObjectList.is_o3_cpu(CPUClass):
    print(f"Configuring O3CPU parameters for {CPUClass.__name__}...")
    try:
        CPUClass.issueWidth = args.issue_width
        print(f"Set {CPUClass.__name__}.issueWidth = {args.issue_width}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'issueWidth'. Error: {e}")
    try:
        CPUClass.decodeWidth = args.decode_width
        print(f"Set {CPUClass.__name__}.decodeWidth = {args.decode_width}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'decodeWidth'. Error: {e}")
    try:
        CPUClass.renameWidth = args.rename_width
        print(f"Set {CPUClass.__name__}.renameWidth = {args.rename_width}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'renameWidth'. Error: {e}")
    try:
        CPUClass.commitWidth = args.commit_width
        print(f"Set {CPUClass.__name__}.commitWidth = {args.commit_width}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'commitWidth'. Error: {e}")
    try:
        CPUClass.fetchWidth = args.fetch_width
        print(f"Set {CPUClass.__name__}.fetchWidth = {args.fetch_width}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'fetchWidth'. Error: {e}")
    try:
        CPUClass.numIntAlu = args.num_int_alu
        print(f"Set {CPUClass.__name__}.numIntAlu = {args.num_int_alu}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'numIntAlu'. Error: {e}")
    try:
        CPUClass.numFpAlu = args.num_fp_alu
        print(f"Set {CPUClass.__name__}.numFpAlu = {args.num_fp_alu}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'numFpAlu'. Error: {e}")
    try:
        CPUClass.iqEntries = args.num_iq_entries
        print(f"Set {CPUClass.__name__}.iqEntries = {args.num_iq_entries}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'iqEntries'. Error: {e}")
    try:
        CPUClass.robEntries = args.num_rob_entries
        print(f"Set {CPUClass.__name__}.robEntries = {args.num_rob_entries}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'robEntries'. Error: {e}")
    try:
        CPUClass.lsqEntries = args.num_lsq_entries
        print(f"Set {CPUClass.__name__}.lsqEntries = {args.num_lsq_entries}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'lsqEntries'. Error: {e}")
    print(f"Finished attempting to configure {CPUClass.__name__} parameters.")

# For MinorCPU, only fetchWidth is commonly configured this way.
elif CPUClass == X86MinorCPU: # Check the class directly
    try:
        CPUClass.fetchWidth = args.fetch_width
        print(f"Set {CPUClass.__name__}.fetchWidth = {args.fetch_width}")
    except AttributeError as e:
        warn(f"'{CPUClass.__name__}' has no direct parameter 'fetchWidth'. Error: {e}")


CPUClass.numThreads = numThreads # This line should be here, after parameter attempts

# Check -- do not allow SMT with multiple CPUs
if args.smt and args.num_cpus > 1:
    fatal("You cannot use SMT with multiple CPUs!")

np = args.num_cpus
mp0_path = multiprocesses[0].executable # Defined here for system.workload later

# Create system object
system = System(
    cpu=[CPUClass(cpu_id=i) for i in range(np)], # Basic CPU creation
    mem_mode=test_mem_mode,
    mem_ranges=[AddrRange(args.mem_size)],
    cache_line_size=args.cacheline_size,
)

if numThreads > 1:
    system.multi_thread = True

# Create a top-level voltage domain
system.voltage_domain = VoltageDomain(voltage=args.sys_voltage)

# Create a source clock for the system and set the clock period
system.clk_domain = SrcClockDomain(
    clock=args.sys_clock, voltage_domain=system.voltage_domain
)

# Create a CPU voltage domain
system.cpu_voltage_domain = VoltageDomain()

# Create a separate clock domain for the CPUs
system.cpu_clk_domain = SrcClockDomain(
    clock=args.cpu_clock, voltage_domain=system.cpu_voltage_domain
)

# If elastic tracing is enabled, then configure the cpu and attach the elastic
# trace probe
if args.elastic_trace_en:
    CpuConfig.config_etrace(CPUClass, system.cpu, args)


# *** REMOVED THE PREVIOUS LOOP TO SET PARAMETERS ON cpu.iew, etc. ***
# Based on 'AttributeError: object X86O3CPU has no attribute 'iew'',
# these sub-components are not directly accessible this way.
# We are now trying to set them on CPUClass itself, before instantiation.


# General CPU configurations (branch predictor, threads)
for cpu in system.cpu: # Still loop for general config like branch predictor
    cpu.clk_domain = system.cpu_clk_domain # This should always be set

    if args.bp_type:
        bpClass = ObjectList.bp_list.get(args.bp_type)
        cpu.branchPred = bpClass()

    if args.indirect_bp_type:
        indirectBPClass = ObjectList.indirect_bp_list.get(
            args.indirect_bp_type
        )
        cpu.branchPred.indirectBranchPred = indirectBPClass()

    # cpu.createThreads() should ideally be handled by CPUClass.numThreads set above
    # and the constructor, but we keep it here for robustness if needed.
    cpu.createThreads()


if ObjectList.is_kvm_cpu(CPUClass) or ObjectList.is_kvm_cpu(FutureClass):
    if buildEnv["USE_X86_ISA"]:
        system.kvm_vm = KvmVM()
        system.m5ops_base = max(0xFFFF0000, Addr(args.mem_size).getValue())
        for process in multiprocesses:
            process.useArchPT = True
            process.kvmInSE = True
    else:
        fatal("KvmCPU can only be used in SE mode with x86")

# Sanity check
if args.simpoint_profile:
    if not ObjectList.is_noncaching_cpu(CPUClass):
        fatal("SimPoint/BPProbe should be done with an atomic cpu")
    if np > 1:
        fatal("SimPoint generation not supported with more than one CPUs")

for i in range(np):
    if args.smt:
        system.cpu[i].workload = multiprocesses
    elif len(multiprocesses) == 1:
        system.cpu[i].workload = multiprocesses[0]
    else:
        system.cpu[i].workload = multiprocesses[i]

    if args.simpoint_profile:
        system.cpu[i].addSimPointProbe(args.simpoint_interval)

    if args.checker:
        system.cpu[i].addCheckerCpu()


if args.ruby:
    Ruby.create_system(args, False, system)
    assert args.num_cpus == len(system.ruby._cpu_ports)

    system.ruby.clk_domain = SrcClockDomain(
        clock=args.ruby_clock, voltage_domain=system.voltage_domain
    )
    for i in range(np):
        ruby_port = system.ruby._cpu_ports[i]

        # Create the interrupt controller and connect its ports to Ruby
        # Note that the interrupt controller is always present but only
        # in x86 does it have message ports that need to be connected
        system.cpu[i].createInterruptController()

        # Connect the cpu's cache ports to Ruby
        ruby_port.connectCpuPorts(system.cpu[i])
else:
    MemClass = Simulation.setMemClass(args)
    system.membus = SystemXBar()
    system.system_port = system.membus.cpu_side_ports
    CacheConfig.config_cache(args, system)
    MemConfig.config_mem(args, system)
    config_filesystem(system, args)

system.workload = SEWorkload.init_compatible(mp0_path)

if args.wait_gdb:
    system.workload.wait_for_remote_gdb = True

root = Root(full_system=False, system=system)
Simulation.run(args, root, system, FutureClass)
