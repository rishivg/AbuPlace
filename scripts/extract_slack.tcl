# extract_slack.tcl — dump per-net worst-slack to JSON from a routed ORFS run.
#
# Reads the post-route ODB + SDC for a design, iterates timing paths to find
# the worst slack on each net, writes JSON: { net_name: slack_ns, ... }.
#
# Inputs (env vars):
#   EXTRACT_ODB         — path to a routed .odb file (e.g. 5_route.odb)
#   EXTRACT_SDC         — path to the design's SDC
#   EXTRACT_LIB_FILES   — space-separated list of liberty files
#   EXTRACT_TECH_LEF    — tech LEF
#   EXTRACT_LEFS        — space-separated additional LEFs (std cells + macros)
#   EXTRACT_OUT_JSON    — output JSON path
#   EXTRACT_PATH_COUNT  — max paths to inspect (default 50000)
#
# Run via:
#   openroad -no_init -exit -metrics extract_slack.json extract_slack.tcl

set odb_path  $::env(EXTRACT_ODB)
set sdc_path  $::env(EXTRACT_SDC)
set tech_lef  $::env(EXTRACT_TECH_LEF)
set lefs      $::env(EXTRACT_LEFS)
set libs      $::env(EXTRACT_LIB_FILES)
set out_json  $::env(EXTRACT_OUT_JSON)
set path_count [expr {[info exists ::env(EXTRACT_PATH_COUNT)] ? $::env(EXTRACT_PATH_COUNT) : 50000}]

puts "extract_slack: loading LEF + liberty + ODB + SDC..."
read_lef $tech_lef
foreach lef [split $lefs] {
    if {[string length $lef] > 0} { read_lef $lef }
}
foreach lib [split $libs] {
    if {[string length $lib] > 0} { read_liberty $lib }
}
read_db $odb_path
read_sdc $sdc_path

# Estimate parasitics from the routed result. -global_routing uses GRT
# results; for a fully-routed design we have real RC stored in SPEF if
# one was written, otherwise fall back to GRT estimate.
puts "extract_slack: estimating parasitics..."
catch { read_spef [file rootname $odb_path].spef } spef_err
if {[string length $spef_err] > 0} {
    puts "extract_slack: no SPEF found ($spef_err); falling back to GRT estimate"
    estimate_parasitics -global_routing
}

# Walk the worst $path_count timing paths. For each path, attribute the
# path's slack to every net it touches (worst-slack wins per net).
puts "extract_slack: walking up to $path_count paths..."
array set net_slack {}

# find_timing_paths returns PathRef objects. -group_path_count caps paths
# per timing group (typically one group "clk_i"), -endpoint_path_count caps
# per endpoint. Combine to bound total work.
set paths [find_timing_paths \
    -path_delay max \
    -group_path_count $path_count \
    -endpoint_path_count 1 \
    -slack_max 0.0 \
    -sort_by_slack]

set n_paths [llength $paths]
puts "extract_slack: found $n_paths violating paths"

set walked 0
foreach path $paths {
    set slack [get_property $path slack]
    if {$slack eq "" || $slack eq "INF"} { continue }
    # OpenSTA's get_property returns slack in the design's declared
    # time_units (1ns for NG45). No conversion needed; values like
    # `-2.05` already mean -2.05 ns.
    set slack_ns $slack

    # Walk the points; each point's pin is on a net. We resolve the net
    # via the pin's full_name → parent instance, then look up the net
    # the pin connects to via ODB's iterm API.
    set points [get_property $path points]
    foreach point $points {
        set pin [get_property $point pin]
        if {$pin eq ""} { continue }
        # ODB API: pin → iterm/bterm → net.
        set iterm [sta::sta_to_db_pin $pin]
        if {$iterm eq "NULL"} {
            # Could be a top-level port (bterm).
            set bterm [sta::sta_to_db_port $pin]
            if {$bterm eq "NULL"} { continue }
            set db_net [$bterm getNet]
        } else {
            set db_net [$iterm getNet]
        }
        if {$db_net eq "NULL"} { continue }
        set net_name [$db_net getName]
        if {![info exists net_slack($net_name)] || $slack_ns < $net_slack($net_name)} {
            set net_slack($net_name) $slack_ns
        }
    }

    incr walked
    if {$walked % 5000 == 0} { puts "extract_slack: walked $walked paths" }
}

# Write JSON.
puts "extract_slack: writing $out_json ([array size net_slack] nets)"
set fp [open $out_json w]
puts $fp "{"
set first 1
foreach name [lsort [array names net_slack]] {
    if {!$first} { puts $fp "," }
    set first 0
    # Quote net names — they can contain [ ] / etc.
    set safe [string map {\\ \\\\ \" \\\"} $name]
    puts -nonewline $fp "  \"$safe\": $net_slack($name)"
}
puts $fp ""
puts $fp "}"
close $fp
puts "extract_slack: done — [array size net_slack] nets with negative slack"

exit 0
