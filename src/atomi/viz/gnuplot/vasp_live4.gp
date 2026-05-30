if (!exists("nfiles")) {
    nfiles = 1
}
if (nfiles < 1) {
    nfiles = 1
}
if (nfiles > 4) {
    nfiles = 4
}

if (!exists("win")) {
    win = 100
}
if (win < 1) {
    win = 100
}
if (!exists("timefile1")) {
    timefile1 = ""
}
if (!exists("timefile2")) {
    timefile2 = ""
}
if (!exists("timefile3")) {
    timefile3 = ""
}
if (!exists("timefile4")) {
    timefile4 = ""
}

set term dumb ansi 160 56
set multiplot layout 2,2 title sprintf("VASP SCF Monitor (log10|dE| + Energy + observed DAV time, window=%d)", int(win))

# ---------- panel 1 ----------
if (nfiles >= 1) {
    file = file1
    timefile = timefile1
    if (!exists("fileshell1")) {
        fileshell1 = file1
    }
    if (!exists("timefileshell1")) {
        timefileshell1 = timefile1
    }
    fileshell = fileshell1
    timefileshell = timefileshell1

    fname_cmd      = "basename -- " . fileshell
    latest_de_cmd  = "awk '/^[[:space:]]*DAV:/{de=$4} END{print (de!=\"\"?de:\"NA\")}' " . fileshell
    latest_E_cmd   = "awk '/^[[:space:]]*DAV:/{e=$3} END{print (e!=\"\"?e:\"NA\")}' " . fileshell
    nstep_cmd      = "awk '/^[[:space:]]*DAV:/{n++} END{print (n>0?n:0)}' " . fileshell
    latest_dt_cmd  = "awk 'NF>=4 && $1 !~ /^#/{dt=$3} /^# state/{base=$3} END{if(dt!=\"\") print sprintf(\"%.1fs\",dt); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefileshell
    mean_dt_cmd    = "awk 'NF>=4 && $1 !~ /^#/{sum+=$3; n++} /^# state/{base=$3} END{if(n>0) print sprintf(\"%.1fs\",sum/n); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefileshell

    fname      = system(fname_cmd)
    latest_de  = system(latest_de_cmd)
    latest_E   = system(latest_E_cmd)
    nsteps_str = system(nstep_cmd)
    latest_dt  = (strlen(timefile) > 0) ? system(latest_dt_cmd) : "NA"
    mean_dt    = (strlen(timefile) > 0) ? system(mean_dt_cmd) : "NA"
    nsteps     = int(nsteps_str)

    xmax = (nsteps > win) ? nsteps : win
    xmin = (nsteps > win) ? (nsteps - win + 1) : 1

    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set title sprintf("%s\nE: %s    dE: %s\nDAV time: latest %s, mean %s", fname, latest_E, latest_de, latest_dt, mean_dt)

    set xlabel ""
    set ylabel "log10(|dE|)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Energy (eV)" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
        "< awk '/^[[:space:]]*DAV:/{c++; v=$4+0; if (v<0) v=-v; if (v>0){print c, log(v)/log(10); p++}} END{if(p==0) print 1,0}' ".fileshell using 1:2 with lines lc rgb "red", \
        "< awk '/^[[:space:]]*DAV:/{c++; if(NF>=3){print c,$3; p++}} END{if(p==0) print 1,0}' ".fileshell using 1:2 axes x1y2 with lines lc rgb "cyan"
}

# ---------- panel 2 ----------
if (nfiles >= 2) {
    file = file2
    timefile = timefile2
    if (!exists("fileshell2")) {
        fileshell2 = file2
    }
    if (!exists("timefileshell2")) {
        timefileshell2 = timefile2
    }
    fileshell = fileshell2
    timefileshell = timefileshell2

    fname_cmd      = "basename -- " . fileshell
    latest_de_cmd  = "awk '/^[[:space:]]*DAV:/{de=$4} END{print (de!=\"\"?de:\"NA\")}' " . fileshell
    latest_E_cmd   = "awk '/^[[:space:]]*DAV:/{e=$3} END{print (e!=\"\"?e:\"NA\")}' " . fileshell
    nstep_cmd      = "awk '/^[[:space:]]*DAV:/{n++} END{print (n>0?n:0)}' " . fileshell
    latest_dt_cmd  = "awk 'NF>=4 && $1 !~ /^#/{dt=$3} /^# state/{base=$3} END{if(dt!=\"\") print sprintf(\"%.1fs\",dt); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefileshell
    mean_dt_cmd    = "awk 'NF>=4 && $1 !~ /^#/{sum+=$3; n++} /^# state/{base=$3} END{if(n>0) print sprintf(\"%.1fs\",sum/n); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefileshell

    fname      = system(fname_cmd)
    latest_de  = system(latest_de_cmd)
    latest_E   = system(latest_E_cmd)
    nsteps_str = system(nstep_cmd)
    latest_dt  = (strlen(timefile) > 0) ? system(latest_dt_cmd) : "NA"
    mean_dt    = (strlen(timefile) > 0) ? system(mean_dt_cmd) : "NA"
    nsteps     = int(nsteps_str)

    xmax = (nsteps > win) ? nsteps : win
    xmin = (nsteps > win) ? (nsteps - win + 1) : 1

    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set title sprintf("%s\nE: %s    dE: %s\nDAV time: latest %s, mean %s", fname, latest_E, latest_de, latest_dt, mean_dt)

    set xlabel ""
    set ylabel "log10(|dE|)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Energy (eV)" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
        "< awk '/^[[:space:]]*DAV:/{c++; v=$4+0; if (v<0) v=-v; if (v>0){print c, log(v)/log(10); p++}} END{if(p==0) print 1,0}' ".fileshell using 1:2 with lines lc rgb "red", \
        "< awk '/^[[:space:]]*DAV:/{c++; if(NF>=3){print c,$3; p++}} END{if(p==0) print 1,0}' ".fileshell using 1:2 axes x1y2 with lines lc rgb "cyan"
}

# ---------- panel 3 ----------
if (nfiles >= 3) {
    file = file3
    timefile = timefile3
    if (!exists("fileshell3")) {
        fileshell3 = file3
    }
    if (!exists("timefileshell3")) {
        timefileshell3 = timefile3
    }
    fileshell = fileshell3
    timefileshell = timefileshell3

    fname_cmd      = "basename -- " . fileshell
    latest_de_cmd  = "awk '/^[[:space:]]*DAV:/{de=$4} END{print (de!=\"\"?de:\"NA\")}' " . fileshell
    latest_E_cmd   = "awk '/^[[:space:]]*DAV:/{e=$3} END{print (e!=\"\"?e:\"NA\")}' " . fileshell
    nstep_cmd      = "awk '/^[[:space:]]*DAV:/{n++} END{print (n>0?n:0)}' " . fileshell
    latest_dt_cmd  = "awk 'NF>=4 && $1 !~ /^#/{dt=$3} /^# state/{base=$3} END{if(dt!=\"\") print sprintf(\"%.1fs\",dt); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefileshell
    mean_dt_cmd    = "awk 'NF>=4 && $1 !~ /^#/{sum+=$3; n++} /^# state/{base=$3} END{if(n>0) print sprintf(\"%.1fs\",sum/n); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefileshell

    fname      = system(fname_cmd)
    latest_de  = system(latest_de_cmd)
    latest_E   = system(latest_E_cmd)
    nsteps_str = system(nstep_cmd)
    latest_dt  = (strlen(timefile) > 0) ? system(latest_dt_cmd) : "NA"
    mean_dt    = (strlen(timefile) > 0) ? system(mean_dt_cmd) : "NA"
    nsteps     = int(nsteps_str)

    xmax = (nsteps > win) ? nsteps : win
    xmin = (nsteps > win) ? (nsteps - win + 1) : 1

    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set title sprintf("%s\nE: %s    dE: %s\nDAV time: latest %s, mean %s", fname, latest_E, latest_de, latest_dt, mean_dt)

    set xlabel "DAV iteration"
    set ylabel "log10(|dE|)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Energy (eV)" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
        "< awk '/^[[:space:]]*DAV:/{c++; v=$4+0; if (v<0) v=-v; if (v>0){print c, log(v)/log(10); p++}} END{if(p==0) print 1,0}' ".fileshell using 1:2 with lines lc rgb "red", \
        "< awk '/^[[:space:]]*DAV:/{c++; if(NF>=3){print c,$3; p++}} END{if(p==0) print 1,0}' ".fileshell using 1:2 axes x1y2 with lines lc rgb "cyan"
}

# ---------- panel 4 ----------
if (nfiles >= 4) {
    file = file4
    timefile = timefile4
    if (!exists("fileshell4")) {
        fileshell4 = file4
    }
    if (!exists("timefileshell4")) {
        timefileshell4 = timefile4
    }
    fileshell = fileshell4
    timefileshell = timefileshell4

    fname_cmd      = "basename -- " . fileshell
    latest_de_cmd  = "awk '/^[[:space:]]*DAV:/{de=$4} END{print (de!=\"\"?de:\"NA\")}' " . fileshell
    latest_E_cmd   = "awk '/^[[:space:]]*DAV:/{e=$3} END{print (e!=\"\"?e:\"NA\")}' " . fileshell
    nstep_cmd      = "awk '/^[[:space:]]*DAV:/{n++} END{print (n>0?n:0)}' " . fileshell
    latest_dt_cmd  = "awk 'NF>=4 && $1 !~ /^#/{dt=$3} /^# state/{base=$3} END{if(dt!=\"\") print sprintf(\"%.1fs\",dt); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefileshell
    mean_dt_cmd    = "awk 'NF>=4 && $1 !~ /^#/{sum+=$3; n++} /^# state/{base=$3} END{if(n>0) print sprintf(\"%.1fs\",sum/n); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefileshell

    fname      = system(fname_cmd)
    latest_de  = system(latest_de_cmd)
    latest_E   = system(latest_E_cmd)
    nsteps_str = system(nstep_cmd)
    latest_dt  = (strlen(timefile) > 0) ? system(latest_dt_cmd) : "NA"
    mean_dt    = (strlen(timefile) > 0) ? system(mean_dt_cmd) : "NA"
    nsteps     = int(nsteps_str)

    xmax = (nsteps > win) ? nsteps : win
    xmin = (nsteps > win) ? (nsteps - win + 1) : 1

    unset label
    unset xrange
    unset y2tics
    set xrange [xmin:xmax]
    set autoscale y
    set autoscale y2

    set title sprintf("%s\nE: %s    dE: %s\nDAV time: latest %s, mean %s", fname, latest_E, latest_de, latest_dt, mean_dt)

    set xlabel "DAV iteration"
    set ylabel "log10(|dE|)" textcolor rgb "red"
    set ytics textcolor rgb "red"

    set y2label "Energy (eV)" textcolor rgb "cyan"
    set y2tics textcolor rgb "cyan"

    set grid
    set key off

    plot \
        "< awk '/^[[:space:]]*DAV:/{c++; v=$4+0; if (v<0) v=-v; if (v>0){print c, log(v)/log(10); p++}} END{if(p==0) print 1,0}' ".fileshell using 1:2 with lines lc rgb "red", \
        "< awk '/^[[:space:]]*DAV:/{c++; if(NF>=3){print c,$3; p++}} END{if(p==0) print 1,0}' ".fileshell using 1:2 axes x1y2 with lines lc rgb "cyan"
}

unset multiplot
