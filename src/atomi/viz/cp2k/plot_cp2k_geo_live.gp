if (!exists("file")) {
    print "Usage: plotcp2k <cp2k_geoopt.log>"
    exit
}

if (!exists("geodat")) {
    print "Missing geodat"
    exit
}

if (!exists("scfdat")) {
    scfdat = ""
}

if (!exists("xyzfile")) {
    xyzfile = ""
}

refresh = 5
set datafile commentschars "#"

if (!exists("helper_py")) {
    helper_py = "cp2k_md_bondtrack.py"
}
bond_dat  = "cp2k_geo_bonds.dat"
bond_meta = "cp2k_geo_bonds.meta"

if (strlen(xyzfile) > 0) {
    helper_cmd = sprintf("python3 '%s' '%s' '%s' '%s' > cp2k_geo_bondtrack.debug 2>&1", helper_py, file, xyzfile, bond_dat)
    system(helper_cmd)
}

system("clear")
set term dumb size 170,58 noenhanced

# -------- latest values --------
fname_cmd         = "basename " . file
latest_gstep_cmd  = "awk 'END{if (NR>0) print $1; else print \"NA\"}' " . geodat
latest_E_cmd      = "awk 'END{if (NR>0) print $2; else print \"NA\"}' " . geodat
latest_gmax_cmd   = "awk 'END{if (NR>0) print $7; else print \"NA\"}' " . geodat
latest_grms_cmd   = "awk 'END{if (NR>0) print $8; else print \"NA\"}' " . geodat
latest_trust_cmd  = "awk 'END{if (NR>0) print $9; else print \"NA\"}' " . geodat

fname        = system(fname_cmd)
latest_gstep = system(latest_gstep_cmd)
latest_E     = system(latest_E_cmd)
latest_gmax  = system(latest_gmax_cmd)
latest_grms  = system(latest_grms_cmd)
latest_trust = system(latest_trust_cmd)

# -------- bond metadata --------
metal_sym = "NA"
coord_n   = "NA"
lig_types = "NA"
if (int(system(sprintf("test -f '%s'; echo $?", bond_meta))) == 0) {
    metal_sym = system("awk -F= '/^metal_symbol=/{print $2}' ".bond_meta)
    coord_n   = system("awk -F= '/^coordination_number=/{print $2}' ".bond_meta)
    lig_types = system("awk -F= '/^ligand_types=/{print $2}' ".bond_meta)
}

# -------- SCF window --------
have_scf = 0
nscf = 0
laststep = -1
if (strlen(scfdat) > 0 && int(system(sprintf("test -s '%s'; echo $?", scfdat))) == 0) {
    stats scfdat using 1 nooutput
    if (STATS_records > 0) {
        have_scf = 1
        laststep = STATS_max
        stats scfdat using (($1==laststep)?$2:1/0) nooutput
        nscf = STATS_records
    }
}

xmax = (nscf > 40) ? nscf : 40
xmin = (nscf > 40) ? (xmax - 39) : 1

# -------- GEO window --------
stats geodat using 1 nooutput
gxmax = (STATS_records > 40) ? STATS_max : 40
gxmin = (STATS_records > 40) ? (gxmax - 39) : 0

unset label
set multiplot layout 3,2 title "CP2K GEO live monitor"

set tmargin 7
set bmargin 3
set lmargin 10
set rmargin 5

set label 1 sprintf("file: %s", fname) at screen 0.02,0.992 left
set label 2 sprintf("latest GEO step: %s", latest_gstep) at screen 0.30,0.992 left
set label 3 sprintf("latest E: %s", latest_E) at screen 0.62,0.992 left
set label 4 sprintf("Metal/CN/Lig: %s / %s / %s", metal_sym, coord_n, lig_types) at screen 0.02,0.965 left
set label 5 sprintf("max grad: %s", latest_gmax) at screen 0.02,0.938 left
set label 6 sprintf("rms grad: %s", latest_grms) at screen 0.28,0.938 left
set label 7 sprintf("trust radius: %s", latest_trust) at screen 0.54,0.938 left

# =========================================================
# Panel 1 — SCF energy
# =========================================================
set grid
set key off
set title "Current GEO step SCF energy"
set xlabel "SCF iteration"
set ylabel "Energy (Ha)"
set xrange [xmin:xmax]
set autoscale y

if (have_scf) {
    plot scfdat using (($1==laststep)?$2:1/0):(($1==laststep)?$5:1/0) \
         with linespoints pt 7 ps 0.25 lw 1.2
} else {
    plot NaN title "no SCF table parsed"
}

# =========================================================
# Panel 2 — SCF convergence
# =========================================================
set grid
set key off
set title "Current GEO step SCF convergence"
set xlabel "SCF iteration"
set ylabel "log10(conv)"
set xrange [xmin:xmax]
set autoscale y

if (have_scf) {
    plot scfdat using (($1==laststep && $4>0)?$2:1/0):(($1==laststep && $4>0)?log($4)/log(10):1/0) \
         with linespoints pt 7 ps 0.25 lw 1.2
} else {
    plot NaN title "no SCF convergence parsed"
}

# =========================================================
# Panel 3 — GEO energy
# =========================================================
set grid
set key off
set title sprintf("Completed GEO energy   trust radius=%s", latest_trust)
set xlabel "GEO step"
set ylabel "Energy (Ha)"
set xrange [gxmin:gxmax]
set autoscale y

plot geodat using 1:2 with lines lw 1.5

# =========================================================
# Panel 4 — gradients
# =========================================================
set grid
set key right
set title "Completed GEO gradients"
set xlabel "GEO step"
set ylabel "log10(gradient)"
set xrange [gxmin:gxmax]
set yrange [-4:0]

plot \
  geodat using 1:(($7>0)?log($7)/log(10):1/0) with lines lw 1.5 title "Max grad", \
  geodat using 1:(($8>0)?log($8)/log(10):1/0) with lines lw 1.5 title "RMS grad", \
  -3.0000 with lines lw 1.2 dt 2 title "target max", \
  -3.1549 with lines lw 1.2 dt 4 title "target rms"

# ============================================================
# PANEL 5 : Individual nearest-shell bond distances
# ============================================================
set title "Nearest-shell metal-ligand distances"
set xlabel "Frame"
set ylabel "Distance (Angstrom)"
set grid
set key right

if (int(system(sprintf("test -f '%s'; echo $?", bond_dat))) == 0) {
    ncols = int(system(sprintf("awk '!/^#/ {print NF; exit}' '%s'", bond_dat)))
    nrows = int(system(sprintf("awk '!/^#/ {n++} END{print n+0}' '%s'", bond_dat)))
    if (ncols >= 5) {
        if (nrows <= 1) {
            set xrange [0.5:1.5]
            plot for [col=5:ncols] bond_dat using 1:col with points title sprintf("d%d", col-4)
        } else {
            set autoscale x
            plot for [col=5:ncols] bond_dat using 1:col with lines title sprintf("d%d", col-4)
        }
    } else {
        plot NaN title "no bond-distance columns"
    }
} else {
    plot NaN title "bond data not available"
}

# ============================================================
# PANEL 6 : Bond summary
# ============================================================
set title "Bond summary"
set xlabel "Frame"
set ylabel "Distance (Angstrom)"
set grid
set key right

if (int(system(sprintf("test -f '%s'; echo $?", bond_dat))) == 0) {
    nrows = int(system(sprintf("awk '!/^#/ {n++} END{print n+0}' '%s'", bond_dat)))
    if (nrows <= 1) {
        set xrange [0.5:1.5]
        plot \
        bond_dat using 1:2 with points title "min", \
        bond_dat using 1:3 with points title "max", \
        bond_dat using 1:4 with points title "mean"
    } else {
        set autoscale x
        plot \
        bond_dat using 1:2 with lines title "min", \
        bond_dat using 1:3 with lines title "max", \
        bond_dat using 1:4 with lines title "mean"
    }
} else {
    plot NaN title "bond data not available"
}

unset multiplot
pause refresh
reread
