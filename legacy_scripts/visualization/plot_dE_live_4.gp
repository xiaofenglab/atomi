if (!exists("nfiles")) nfiles = 1
if (nfiles < 1) nfiles = 1
if (nfiles > 4) nfiles = 4

if (!exists("win")) win = 100
if (win < 1) win = 100

set term dumb ansi 160 48
set multiplot layout 2,2 title sprintf("VASP SCF Monitor (log10|dE| + Energy, window=%d)", int(win))

# ---------- panel 1 ----------
if (nfiles >= 1) {
    file = file1

    fname_cmd      = "basename " . file
    latest_de_cmd  = "awk '/^DAV:/{de=$4} END{print (de!=\"\"?de:\"NA\")}' " . file
    latest_E_cmd   = "awk '/^DAV:/{e=$3} END{print (e!=\"\"?e:\"NA\")}' " . file
    nstep_cmd      = "awk '/^DAV:/{n++} END{print (n>0?n:0)}' " . file

    fname      = system(fname_cmd)
    latest_de  = system(latest_de_cmd)
    latest_E   = system(latest_E_cmd)
    nsteps_str = system(nstep_cmd)
    nsteps     = int(nsteps_str)

    xmax = (nsteps > win) ? nsteps : win
    xmin = (nsteps > win) ? (nsteps - win + 1) : 1

    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set title sprintf("%s", fname)
    set label 1 sprintf("E: %s", latest_E)   at screen 0.11,0.965 left textcolor rgb "cyan"
    set label 2 sprintf("dE: %s", latest_de) at screen 0.24,0.965 left textcolor rgb "red"

    set xlabel ""
    set ylabel "log10(|dE|)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Energy (eV)" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
        "< awk '/^DAV:/{c++; v=$4+0; if (v<0) v=-v; if (v>0) print c, log(v)/log(10)}' ".file using 1:2 with lines lc rgb "red", \
        "< awk '/^DAV:/{c++; print c,$3}' ".file using 1:2 axes x1y2 with lines lc rgb "cyan"
}

# ---------- panel 2 ----------
if (nfiles >= 2) {
    file = file2

    fname_cmd      = "basename " . file
    latest_de_cmd  = "awk '/^DAV:/{de=$4} END{print (de!=\"\"?de:\"NA\")}' " . file
    latest_E_cmd   = "awk '/^DAV:/{e=$3} END{print (e!=\"\"?e:\"NA\")}' " . file
    nstep_cmd      = "awk '/^DAV:/{n++} END{print (n>0?n:0)}' " . file

    fname      = system(fname_cmd)
    latest_de  = system(latest_de_cmd)
    latest_E   = system(latest_E_cmd)
    nsteps_str = system(nstep_cmd)
    nsteps     = int(nsteps_str)

    xmax = (nsteps > win) ? nsteps : win
    xmin = (nsteps > win) ? (nsteps - win + 1) : 1

    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set title sprintf("%s", fname)
    set label 1 sprintf("E: %s", latest_E)   at screen 0.61,0.965 left textcolor rgb "cyan"
    set label 2 sprintf("dE: %s", latest_de) at screen 0.74,0.965 left textcolor rgb "red"

    set xlabel ""
    set ylabel "log10(|dE|)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Energy (eV)" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
        "< awk '/^DAV:/{c++; v=$4+0; if (v<0) v=-v; if (v>0) print c, log(v)/log(10)}' ".file using 1:2 with lines lc rgb "red", \
        "< awk '/^DAV:/{c++; print c,$3}' ".file using 1:2 axes x1y2 with lines lc rgb "cyan"
}

# ---------- panel 3 ----------
if (nfiles >= 3) {
    file = file3

    fname_cmd      = "basename " . file
    latest_de_cmd  = "awk '/^DAV:/{de=$4} END{print (de!=\"\"?de:\"NA\")}' " . file
    latest_E_cmd   = "awk '/^DAV:/{e=$3} END{print (e!=\"\"?e:\"NA\")}' " . file
    nstep_cmd      = "awk '/^DAV:/{n++} END{print (n>0?n:0)}' " . file

    fname      = system(fname_cmd)
    latest_de  = system(latest_de_cmd)
    latest_E   = system(latest_E_cmd)
    nsteps_str = system(nstep_cmd)
    nsteps     = int(nsteps_str)

    xmax = (nsteps > win) ? nsteps : win
    xmin = (nsteps > win) ? (nsteps - win + 1) : 1

    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set title sprintf("%s", fname)
    set label 1 sprintf("E: %s", latest_E)   at screen 0.11,0.485 left textcolor rgb "cyan"
    set label 2 sprintf("dE: %s", latest_de) at screen 0.24,0.485 left textcolor rgb "red"

    set xlabel "DAV iteration"
    set ylabel "log10(|dE|)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Energy (eV)" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
        "< awk '/^DAV:/{c++; v=$4+0; if (v<0) v=-v; if (v>0) print c, log(v)/log(10)}' ".file using 1:2 with lines lc rgb "red", \
        "< awk '/^DAV:/{c++; print c,$3}' ".file using 1:2 axes x1y2 with lines lc rgb "cyan"
}

# ---------- panel 4 ----------
if (nfiles >= 4) {
    file = file4

    fname_cmd      = "basename " . file
    latest_de_cmd  = "awk '/^DAV:/{de=$4} END{print (de!=\"\"?de:\"NA\")}' " . file
    latest_E_cmd   = "awk '/^DAV:/{e=$3} END{print (e!=\"\"?e:\"NA\")}' " . file
    nstep_cmd      = "awk '/^DAV:/{n++} END{print (n>0?n:0)}' " . file

    fname      = system(fname_cmd)
    latest_de  = system(latest_de_cmd)
    latest_E   = system(latest_E_cmd)
    nsteps_str = system(nstep_cmd)
    nsteps     = int(nsteps_str)

    xmax = (nsteps > win) ? nsteps : win
    xmin = (nsteps > win) ? (nsteps - win + 1) : 1

    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set title sprintf("%s", fname)
    set label 1 sprintf("E: %s", latest_E)   at screen 0.61,0.485 left textcolor rgb "cyan"
    set label 2 sprintf("dE: %s", latest_de) at screen 0.74,0.485 left textcolor rgb "red"

    set xlabel "DAV iteration"
    set ylabel "log10(|dE|)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Energy (eV)" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
        "< awk '/^DAV:/{c++; v=$4+0; if (v<0) v=-v; if (v>0) print c, log(v)/log(10)}' ".file using 1:2 with lines lc rgb "red", \
        "< awk '/^DAV:/{c++; print c,$3}' ".file using 1:2 axes x1y2 with lines lc rgb "cyan"
}

unset multiplot