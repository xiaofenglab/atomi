# ------------------------------------------------------------
# plot_lammps_thermo.gp
#
# Terminal LAMMPS thermo monitor
# - 2x2 square ASCII plots
# - colored lines
# - works during running simulations
# - parses thermo blocks starting with "Step"
#
# Usage:
#   gnuplot -e "file='log.lammps'; win=40" lammps_thermo.gp
# ------------------------------------------------------------

if (!exists("file")) {
    print "Usage: gnuplot -e \"file='logfile'\" plot_lammps_thermo.gp"
    exit
}

set term dumb ansi 120 40

if (!exists("win")) win = 40
if (win < 1) win = 40

# Extract numeric thermo rows
base = "awk 'BEGIN{f=0} /^ *Step/{f=1; next} f && NF>0 && $1+0==$1 {print} /^Loop time/{f=0}' ".file

# Count rows
ncmd = "awk 'BEGIN{f=0;n=0} /^ *Step/{f=1; next} f && NF>0 && $1+0==$1 {n++} /^Loop time/{f=0} END{print n}' ".file
nsteps = int(system(ncmd))

if (nsteps < 1) {
    print "No thermo data detected."
    exit
}

xmin = (nsteps > win) ? nsteps - win + 1 : 1
xmax = (nsteps > win) ? nsteps : win

print "===================================================="
print sprintf("LAMMPS log : %s", file)
print sprintf("Thermo pts : %d", nsteps)
print "===================================================="

set xrange [xmin:xmax]
set autoscale y
set key off
set grid

set multiplot layout 2,2 title sprintf("LAMMPS Thermo Monitor (%s)", file)

# ------------------------------------------------
# Temperature
# ------------------------------------------------
set title "Temperature"
set xlabel ""
set ylabel "K"
plot sprintf("< %s", base) using 0:2 with lines lc rgb "red"

# ------------------------------------------------
# Pressure
# ------------------------------------------------
set title "Pressure"
set xlabel ""
set ylabel "bar"
plot sprintf("< %s", base) using 0:5 with lines lc rgb "blue"

# ------------------------------------------------
# Volume
# ------------------------------------------------
set title "Volume"
set xlabel ""
set ylabel "A^3"
plot sprintf("< %s", base) using 0:6 with lines lc rgb "green"

# ------------------------------------------------
# Potential Energy
# ------------------------------------------------
set title "PotEng"
set xlabel "thermo index"
set ylabel "eV"
plot sprintf("< %s", base) using 0:3 with lines lc rgb "yellow"

unset multiplot

print "\nDone.\n"
