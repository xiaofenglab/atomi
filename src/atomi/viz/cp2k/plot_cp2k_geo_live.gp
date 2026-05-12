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
if (!exists("track_atom")) {
    track_atom = 0
}

refresh = 5
set datafile commentschars "#"

if (!exists("helper_py")) {
    helper_py = "cp2k_md_bondtrack.py"
}
bond_dat  = "cp2k_geo_bonds.dat"
bond_meta = "cp2k_geo_bonds.meta"

if (strlen(xyzfile) > 0) {
    helper_cmd = sprintf("rm -f '%s' '%s'; python3 '%s' '%s' '%s' '%s' '%d' > cp2k_geo_bondtrack.debug 2>&1", bond_dat, bond_meta, helper_py, file, xyzfile, bond_dat, int(track_atom))
    system(helper_cmd)
}

system("clear")
set term dumb ansi size 170,58 noenhanced

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
dlabel1 = "d1"
dlabel2 = "d2"
dlabel3 = "d3"
dlabel4 = "d4"
dlabel5 = "d5"
dlabel6 = ""
if (int(system(sprintf("test -f '%s'; echo $?", bond_meta))) == 0) {
    metal_sym = system("awk -F= '/^metal_symbol=/{print $2}' ".bond_meta)
    coord_n   = system("awk -F= '/^coordination_number=/{print $2}' ".bond_meta)
    lig_types = system("awk -F= '/^ligand_types=/{print $2}' ".bond_meta)
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[1]}' ".bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel1 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[2]}' ".bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel2 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[3]}' ".bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel3 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[4]}' ".bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel4 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[5]}' ".bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel5 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[6]}' ".bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel6 = tmp_label
    }
}
if (int(system(sprintf("test -s '%s'; echo $?", bond_dat))) == 0) {
    if (strlen(dlabel1) == 0 || (dlabel1 eq "d1")) {
        tmp_label = system("awk '/^#/ {print $6; exit}' ".bond_dat)
        if (strlen(tmp_label) > 0) {
            dlabel1 = tmp_label
        }
    }
    if (strlen(dlabel2) == 0 || (dlabel2 eq "d2")) {
        tmp_label = system("awk '/^#/ {print $7; exit}' ".bond_dat)
        if (strlen(tmp_label) > 0) {
            dlabel2 = tmp_label
        }
    }
    if (strlen(dlabel3) == 0 || (dlabel3 eq "d3")) {
        tmp_label = system("awk '/^#/ {print $8; exit}' ".bond_dat)
        if (strlen(tmp_label) > 0) {
            dlabel3 = tmp_label
        }
    }
    if (strlen(dlabel4) == 0 || (dlabel4 eq "d4")) {
        tmp_label = system("awk '/^#/ {print $9; exit}' ".bond_dat)
        if (strlen(tmp_label) > 0) {
            dlabel4 = tmp_label
        }
    }
    if (strlen(dlabel5) == 0 || (dlabel5 eq "d5")) {
        tmp_label = system("awk '/^#/ {print $10; exit}' ".bond_dat)
        if (strlen(tmp_label) > 0) {
            dlabel5 = tmp_label
        }
    }
    if (strlen(dlabel6) == 0) {
        tmp_label = system("awk '/^#/ {print $11; exit}' ".bond_dat)
        if (strlen(tmp_label) > 0) {
            dlabel6 = tmp_label
        }
    }
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
         with linespoints pt 7 ps 0.25 lw 1.2 lc rgb "cyan"
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
         with linespoints pt 7 ps 0.25 lw 1.2 lc rgb "magenta"
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

plot geodat using 1:2 with lines lw 1.5 lc rgb "cyan"

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
  geodat using 1:(($7>0)?log($7)/log(10):1/0) with lines lw 1.5 lc rgb "red" title "Max grad", \
  geodat using 1:(($8>0)?log($8)/log(10):1/0) with lines lw 1.5 lc rgb "magenta" title "RMS grad", \
  -3.0000 with lines lw 1.2 lc rgb "green" dt 2 title "target max", \
  -3.1549 with lines lw 1.2 lc rgb "cyan" dt 4 title "target rms"

# ============================================================
# PANEL 5 : Individual nearest-shell bond distances
# ============================================================
set title "Nearest-shell metal-ligand distances"
set xlabel "Frame"
set ylabel "Distance (Angstrom)" offset -2,0
set grid
set key outside right top
set rmargin 18

if (int(system(sprintf("test -f '%s'; echo $?", bond_dat))) == 0) {
    ncols = int(system(sprintf("awk '!/^#/ {print NF; exit}' '%s'", bond_dat)))
    nrows = int(system(sprintf("awk '!/^#/ {n++} END{print n+0}' '%s'", bond_dat)))
    if (ncols >= 5) {
        if (nrows <= 1) {
            set xrange [0.5:1.5]
            plot \
            bond_dat using 1:(ncols>=5 ? column(5) : 1/0) with points pt 7 ps 0.35 lc 5 title dlabel1, \
            bond_dat using 1:(ncols>=6 ? column(6) : 1/0) with points pt 7 ps 0.35 lc 6 title dlabel2, \
            bond_dat using 1:(ncols>=7 ? column(7) : 1/0) with points pt 7 ps 0.35 lc 7 title dlabel3, \
            bond_dat using 1:(ncols>=8 ? column(8) : 1/0) with points pt 7 ps 0.35 lc 8 title dlabel4, \
            bond_dat using 1:(ncols>=9 ? column(9) : 1/0) with points pt 7 ps 0.35 lc 9 title dlabel5, \
            bond_dat using 1:((ncols>=10 && strlen(dlabel6)>0) ? column(10) : 1/0) with points pt 7 ps 0.35 lc 10 title dlabel6
        } else {
            set autoscale x
            plot \
            bond_dat using 1:(ncols>=5 ? column(5) : 1/0) with lines lw 1.2 lc 5 title dlabel1, \
            bond_dat using 1:(ncols>=6 ? column(6) : 1/0) with lines lw 1.2 lc 6 title dlabel2, \
            bond_dat using 1:(ncols>=7 ? column(7) : 1/0) with lines lw 1.2 lc 7 title dlabel3, \
            bond_dat using 1:(ncols>=8 ? column(8) : 1/0) with lines lw 1.2 lc 8 title dlabel4, \
            bond_dat using 1:(ncols>=9 ? column(9) : 1/0) with lines lw 1.2 lc 9 title dlabel5, \
            bond_dat using 1:((ncols>=10 && strlen(dlabel6)>0) ? column(10) : 1/0) with lines lw 1.2 lc 10 title dlabel6
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
set ylabel "Distance (Angstrom)" offset -2,0
set grid
set key outside right top
set rmargin 18

if (int(system(sprintf("test -f '%s'; echo $?", bond_dat))) == 0) {
    nrows = int(system(sprintf("awk '!/^#/ {n++} END{print n+0}' '%s'", bond_dat)))
    if (nrows <= 1) {
        set xrange [0.5:1.5]
        plot \
        bond_dat using 1:2 with points pt 7 ps 0.4 lc rgb "blue" title "min", \
        bond_dat using 1:3 with points pt 7 ps 0.4 lc rgb "red" title "max", \
        bond_dat using 1:4 with points pt 7 ps 0.4 lc rgb "green" title "mean"
    } else {
        set autoscale x
        plot \
        bond_dat using 1:2 with lines lw 1.4 lc rgb "blue" title "min", \
        bond_dat using 1:3 with lines lw 1.4 lc rgb "red" title "max", \
        bond_dat using 1:4 with lines lw 1.4 lc rgb "green" title "mean"
    }
} else {
    plot NaN title "bond data not available"
}

unset multiplot
pause refresh
reread
