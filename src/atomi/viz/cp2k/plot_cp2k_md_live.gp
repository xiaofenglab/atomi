# plot_cp2k_md_live.gp
# Usage:
#   gnuplot -e "file='cp2k.log'; xyzfile='traj.xyz'; refresh=15; win=300" plot_cp2k_md_live.gp
#   gnuplot -e "file='cp2k.log'; refresh=15; win=300" plot_cp2k_md_live.gp

if (!exists("file")) {
    print "Usage: gnuplot -e \"file='cp2k.log'; xyzfile='traj.xyz'; refresh=15; win=300\" plot_cp2k_md_live.gp"
    exit
}

if (!exists("xyzfile")) {
    xyzfile = ""
}

if (!exists("refresh")) refresh = 15
if (!exists("win"))     win = 300
if (win < 1)            win = 300
if (!exists("track_atom")) track_atom = 0
track_atom_meta = (int(track_atom) > 0) ? sprintf("%d", int(track_atom)) : "NA"

if (!exists("helper_py")) {
    helper_py = "cp2k_md_bondtrack.py"
}
bond_dat  = "cp2k_md_bonds.dat"
bond_meta = "cp2k_md_bonds.meta"

if (!exists("eta_py")) {
    eta_py = "cp2k_md_eta.py"
}
eta_hist = "cp2k_md_eta.hist"
eta_tmp  = "cp2k_md_eta.tmp"
inpfile  = system("ls *.inp 2>/dev/null | head -n 1")

live_dat  = "cp2k_md_live.dat"
live_meta = "cp2k_md_live.meta"

# ------------------------------------------------------------
# Rebuild parsed CP2K live table in ONE pass
# Columns in live_dat:
#   1 step
#   2 time_fs
#   3 temp_K
#   4 pot_Ha
#   5 kin_Ha
#   6 cons_Ha
#   7 scf_steps
# ------------------------------------------------------------
build_live_cmd = \
"awk '/MD\\| Step number/ {s=$NF} " . \
"/MD\\| Time \\[fs\\]/ {t=$NF} " . \
"/MD\\| Temperature \\[K\\]/ {temp=$(NF-1)} " . \
"/MD\\| Potential energy \\[hartree\\]/ {pot=$(NF-1)} " . \
"/MD\\| Kinetic energy \\[hartree\\]/ {kin=$(NF-1)} " . \
"/MD\\| Conserved quantity \\[hartree\\]/ { " . \
"cons=$NF; " . \
"if (s!=\"\") { " . \
"scf=(last_scf!=\"\" ? last_scf : \"NaN\"); " . \
"print s, t, temp, pot, kin, cons, scf; " . \
"} " . \
"} " . \
"/\\*\\*\\* SCF run converged in/ {last_scf=$(NF-2)} " . \
"END { " . \
"if (s!=\"\") { " . \
"print \"latest_step=\" s > \"" . live_meta . "\"; " . \
"print \"latest_time=\" t >> \"" . live_meta . "\"; " . \
"print \"latest_temp=\" temp >> \"" . live_meta . "\"; " . \
"print \"latest_pot=\" pot >> \"" . live_meta . "\"; " . \
"print \"latest_kin=\" kin >> \"" . live_meta . "\"; " . \
"print \"latest_cons=\" cons >> \"" . live_meta . "\"; " . \
"print \"latest_scf=\" (last_scf!=\"\" ? last_scf : \"NA\") >> \"" . live_meta . "\"; " . \
"} " . \
"}' '" . file . "' > '" . live_dat . "'"
system(build_live_cmd)

# ------------------------------------------------------------
# Rebuild helper outputs only if source files changed
# ------------------------------------------------------------
if (strlen(xyzfile) > 0) {
    bond_update_cmd = \
    "if [ ! -f '" . bond_dat . "' ] || [ '" . helper_py . "' -nt '" . bond_dat . "' ] || " . \
    "[ '" . file . "' -nt '" . bond_dat . "' ] || [ '" . xyzfile . "' -nt '" . bond_dat . "' ] || " . \
    "[ \"$(awk -F= '/^track_atom=/{print $2}' '" . bond_meta . "' 2>/dev/null)\" != '" . track_atom_meta . "' ]; then " . \
    "rm -f '" . bond_dat . "' '" . bond_meta . "'; " . \
    "python3 '" . helper_py . "' '" . file . "' '" . xyzfile . "' '" . bond_dat . "' '" . sprintf("%d", int(track_atom)) . "' > cp2k_md_bondtrack.debug 2>&1; fi"
    system(bond_update_cmd)
}

if (strlen(inpfile) > 0) {
    eta_update_cmd = \
    "if [ ! -f '" . eta_tmp . "' ] || [ '" . file . "' -nt '" . eta_tmp . "' ] || [ '" . inpfile . "' -nt '" . eta_tmp . "' ]; then " . \
    "python3 '" . eta_py . "' '" . file . "' '" . inpfile . "' '" . eta_hist . "' > '" . eta_tmp . "' 2>/dev/null; fi"
    system(eta_update_cmd)
}

system("clear")

set term dumb ansi size 170,64 noenhanced
set multiplot layout 3,2 title sprintf("CP2K AIMD Live Monitor (window=%d, refresh=%ds)", int(win), int(refresh))

set tmargin 8
set bmargin 3
set lmargin 10
set rmargin 5

fname_cmd = "basename " . file
fname = system(fname_cmd)

# -------- latest summary values from parsed meta --------
latest_step = "NA"
latest_time = "NA"
latest_temp = "NA"
latest_pot  = "NA"
latest_kin  = "NA"
latest_cons = "NA"
latest_scf  = "NA"

test_live_meta_cmd = "test -f '" . live_meta . "'; echo $?"
if (int(system(test_live_meta_cmd)) == 0) {
    latest_step = system("awk -F= '/^latest_step=/{print $2}' " . live_meta)
    latest_time = system("awk -F= '/^latest_time=/{print $2}' " . live_meta)
    latest_temp = system("awk -F= '/^latest_temp=/{print $2}' " . live_meta)
    latest_pot  = system("awk -F= '/^latest_pot=/{print $2}' " . live_meta)
    latest_kin  = system("awk -F= '/^latest_kin=/{print $2}' " . live_meta)
    latest_cons = system("awk -F= '/^latest_cons=/{print $2}' " . live_meta)
    latest_scf  = system("awk -F= '/^latest_scf=/{print $2}' " . live_meta)
}

# -------- latest summary values from bond helper --------
metal_sym = "NA"
coord_n   = "NA"
lig_types = "NA"
summary_label = "displayed"
dlabel1 = "d1"
dlabel2 = "d2"
dlabel3 = "d3"
dlabel4 = "d4"
dlabel5 = "d5"
dlabel6 = ""

test_bond_meta_cmd = "test -f '" . bond_meta . "'; echo $?"
if (int(system(test_bond_meta_cmd)) == 0) {
    metal_sym = system("awk -F= '/^metal_symbol=/{print $2}' " . bond_meta)
    coord_n   = system("awk -F= '/^coordination_number=/{print $2}' " . bond_meta)
    lig_types = system("awk -F= '/^ligand_types=/{print $2}' " . bond_meta)
    tmp_label = system("awk -F= '/^summary_label=/{print $2}' " . bond_meta)
    if (strlen(tmp_label) > 0) {
        summary_label = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[1]}' " . bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel1 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[2]}' " . bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel2 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[3]}' " . bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel3 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[4]}' " . bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel4 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[5]}' " . bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel5 = tmp_label
    }
    tmp_label = system("awk -F= '/^distance_labels=/{split($2,a,\",\"); print a[6]}' " . bond_meta)
    if (strlen(tmp_label) > 0) {
        dlabel6 = tmp_label
    }
}
if (int(system("test -s '" . bond_dat . "'; echo $?")) == 0) {
    if (strlen(dlabel1) == 0 || (dlabel1 eq "d1")) {
        tmp_label = system("awk '/^#/ {print $6; exit}' '" . bond_dat . "'")
        if (strlen(tmp_label) > 0) {
            dlabel1 = tmp_label
        }
    }
    if (strlen(dlabel2) == 0 || (dlabel2 eq "d2")) {
        tmp_label = system("awk '/^#/ {print $7; exit}' '" . bond_dat . "'")
        if (strlen(tmp_label) > 0) {
            dlabel2 = tmp_label
        }
    }
    if (strlen(dlabel3) == 0 || (dlabel3 eq "d3")) {
        tmp_label = system("awk '/^#/ {print $8; exit}' '" . bond_dat . "'")
        if (strlen(tmp_label) > 0) {
            dlabel3 = tmp_label
        }
    }
    if (strlen(dlabel4) == 0 || (dlabel4 eq "d4")) {
        tmp_label = system("awk '/^#/ {print $9; exit}' '" . bond_dat . "'")
        if (strlen(tmp_label) > 0) {
            dlabel4 = tmp_label
        }
    }
    if (strlen(dlabel5) == 0 || (dlabel5 eq "d5")) {
        tmp_label = system("awk '/^#/ {print $10; exit}' '" . bond_dat . "'")
        if (strlen(tmp_label) > 0) {
            dlabel5 = tmp_label
        }
    }
    if (strlen(dlabel6) == 0) {
        tmp_label = system("awk '/^#/ {print $11; exit}' '" . bond_dat . "'")
        if (strlen(tmp_label) > 0) {
            dlabel6 = tmp_label
        }
    }
}

# -------- ETA summary --------
latest_mps    = "NA"
latest_spm    = "NA"
latest_remain = "NA"
latest_eta    = "NA"

test_eta_tmp_cmd = "test -f '" . eta_tmp . "'; echo $?"
if (int(system(test_eta_tmp_cmd)) == 0) {
    latest_mps    = system("awk -F= '/^min_per_step=/{print $2}' " . eta_tmp)
    latest_spm    = system("awk -F= '/^steps_per_min=/{print $2}' " . eta_tmp)
    latest_remain = system("awk -F= '/^remaining_steps=/{print $2}' " . eta_tmp)
    latest_eta    = system("awk -F= '/^eta_hms=/{print $2}' " . eta_tmp)
}

# -------- x-window handling --------
nrows_cmd = "awk 'END{print NR+0}' '" . live_dat . "'"
nrows_str = system(nrows_cmd)
nrows = int(nrows_str)

xmax = (nrows > win) ? nrows : win
xmin = (nrows > win) ? (nrows - win + 1) : 1

unset label

set label 1 sprintf("File: %s", fname) at screen 0.02,0.995 left
set label 2 sprintf("Step: %s", latest_step) at screen 0.30,0.995 left
set label 3 sprintf("Time(fs): %s", latest_time) at screen 0.46,0.995 left
set label 4 sprintf("Metal/CN/Lig: %s / %s / %s", metal_sym, coord_n, lig_types) at screen 0.66,0.995 left

set label 5 sprintf("T(K): %s", latest_temp) at screen 0.02,0.970 left
set label 6 sprintf("Pot(Ha): %s", latest_pot) at screen 0.20,0.970 left
set label 7 sprintf("Kin(Ha): %s", latest_kin) at screen 0.44,0.970 left
set label 8 sprintf("Cons(Ha): %s", latest_cons) at screen 0.66,0.970 left

set label 9  sprintf("SCF: %s", latest_scf) at screen 0.02,0.945 left
set label 10 sprintf("min/step: %s", latest_mps) at screen 0.18,0.945 left
set label 11 sprintf("step/min: %s", latest_spm) at screen 0.38,0.945 left
set label 12 sprintf("remain: %s", latest_remain) at screen 0.58,0.945 left
set label 13 sprintf("ETA: %s", latest_eta) at screen 0.82,0.945 right

# ============================================================
# PANEL 1 : Temperature
# ============================================================
set title "Temperature"
set xlabel "MD record"
set ylabel "T (K)"
set grid
set key off
set xrange [xmin:xmax]
set autoscale y
plot live_dat using 0:3 with lines lc rgb "red" lw 1.4

# ============================================================
# PANEL 2 : Potential / Conserved Energy
# ============================================================
set title "Potential / Conserved Energy"
set xlabel "MD record"
set ylabel "Energy (Ha)"
set grid
set key right
set xrange [xmin:xmax]
set autoscale y
plot \
live_dat using 0:4 with lines lc rgb "cyan" lw 1.4 title "Potential", \
live_dat using 0:6 with lines lc rgb "yellow" lw 1.4 title "Conserved"

# ============================================================
# PANEL 3 : Kinetic Energy
# ============================================================
set title "Kinetic Energy"
set xlabel "MD record"
set ylabel "Kinetic (Ha)"
set grid
set key off
set xrange [xmin:xmax]
set autoscale y
plot live_dat using 0:5 with lines lc rgb "magenta" lw 1.4

# ============================================================
# PANEL 4 : SCF effort
# ============================================================
set title "SCF effort"
set xlabel "MD record"
set ylabel "SCF steps"
set grid
set key off
set xrange [xmin:xmax]
set autoscale y
plot live_dat using 0:7 with lines lc rgb "green" lw 1.4

# ============================================================
# PANEL 5 : Individual nearest-shell bond distances
# ============================================================
unset ylabel
set title "Nearest-shell metal-ligand distances\nDistance (Angstrom)"
set xlabel "Frame"
set grid
set key outside right top
set rmargin 18

have_bonds = 0
ncols = 0
bond_rows = 0
if (int(system("test -s '" . bond_dat . "'; echo $?")) == 0) {
    have_bonds = 1
    ncols_cmd = "awk '!/^#/ {print NF; exit}' '" . bond_dat . "'"
    bond_rows_cmd = "awk '!/^#/ {n++} END{print n+0}' '" . bond_dat . "'"
    ncols = int(system(ncols_cmd))
    bond_rows = int(system(bond_rows_cmd))
}

bxmax = (bond_rows > win) ? bond_rows : win
bxmin = (bond_rows > win) ? (bond_rows - win + 1) : 1

if (have_bonds && ncols >= 5) {
    set xrange [bxmin:bxmax]
    if (bond_rows <= 1) {
        plot \
        bond_dat using 1:(ncols>=5 ? column(5) : 1/0) with points pt 7 ps 0.35 lc 5 title dlabel1, \
        bond_dat using 1:(ncols>=6 ? column(6) : 1/0) with points pt 7 ps 0.35 lc 6 title dlabel2, \
        bond_dat using 1:(ncols>=7 ? column(7) : 1/0) with points pt 7 ps 0.35 lc 7 title dlabel3, \
        bond_dat using 1:(ncols>=8 ? column(8) : 1/0) with points pt 7 ps 0.35 lc 8 title dlabel4, \
        bond_dat using 1:(ncols>=9 ? column(9) : 1/0) with points pt 7 ps 0.35 lc 9 title dlabel5, \
        bond_dat using 1:((ncols>=10 && strlen(dlabel6)>0) ? column(10) : 1/0) with points pt 7 ps 0.35 lc 10 title dlabel6
    } else {
        plot \
        bond_dat using 1:(ncols>=5 ? column(5) : 1/0) with lines lw 1.2 lc 5 title dlabel1, \
        bond_dat using 1:(ncols>=6 ? column(6) : 1/0) with lines lw 1.2 lc 6 title dlabel2, \
        bond_dat using 1:(ncols>=7 ? column(7) : 1/0) with lines lw 1.2 lc 7 title dlabel3, \
        bond_dat using 1:(ncols>=8 ? column(8) : 1/0) with lines lw 1.2 lc 8 title dlabel4, \
        bond_dat using 1:(ncols>=9 ? column(9) : 1/0) with lines lw 1.2 lc 9 title dlabel5, \
        bond_dat using 1:((ncols>=10 && strlen(dlabel6)>0) ? column(10) : 1/0) with lines lw 1.2 lc 10 title dlabel6
    }
} else {
    plot NaN title "no bond-distance data"
}

# ============================================================
# PANEL 6 : Bond summary
# ============================================================
set title sprintf("Bond summary (%s)", summary_label)
set xlabel "Frame"
set ylabel "Distance (Angstrom)" offset -2,0
set grid
set key outside right top
set rmargin 18
if (have_bonds) {
    set xrange [bxmin:bxmax]
}

if (!have_bonds) {
    plot NaN title "no bond-distance data"
} else {
    if (bond_rows <= 1) {
        plot \
        bond_dat using 1:2 with points pt 7 ps 0.4 lc rgb "blue" title "min", \
        bond_dat using 1:3 with points pt 7 ps 0.4 lc rgb "red" title "max", \
        bond_dat using 1:4 with points pt 7 ps 0.4 lc rgb "green" title "mean"
    } else {
        plot \
        bond_dat using 1:2 with lines lw 1.4 lc rgb "blue" title "min", \
        bond_dat using 1:3 with lines lw 1.4 lc rgb "red" title "max", \
        bond_dat using 1:4 with lines lw 1.4 lc rgb "green" title "mean"
    }
}

unset multiplot
pause refresh
reread
