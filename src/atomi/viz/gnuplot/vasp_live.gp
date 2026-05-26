if (!exists("file")) {
    print "Usage: plotvasp <vasp.out> [window_steps]"
    exit
}

if (!exists("win")) {
    win = 100
}
if (win < 1) {
    win = 100
}
if (!exists("timefile")) {
    timefile = ""
}

set term dumb ansi 140 56
set multiplot layout 3,1 title sprintf("VASP SCF Monitor (window=%d)", int(win))

fname_cmd      = "basename " . file
latest_de_cmd  = "awk '/^DAV:/{de=$4} END{print (de!=\"\"?de:\"NA\")}' " . file
latest_rms_cmd = "awk '/^DAV:/{r=$7} END{print (r!=\"\"?r:\"NA\")}' " . file
latest_E_cmd   = "awk '/^DAV:/{e=$3} END{print (e!=\"\"?e:\"NA\")}' " . file
nstep_cmd      = "awk '/^DAV:/{n++} END{print (n>0?n:0)}' " . file
latest_dt_cmd  = "awk 'NF>=4 && $1 !~ /^#/{dt=$3} /^# state/{base=$3} END{if(dt!=\"\") print sprintf(\"%.1fs\",dt); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefile
mean_dt_cmd    = "awk 'NF>=4 && $1 !~ /^#/{sum+=$3; n++} /^# state/{base=$3} END{if(n>0) print sprintf(\"%.1fs\",sum/n); else if(base!=\"\") print \"waiting>DAV\" base; else print \"waiting\"}' " . timefile

fname      = system(fname_cmd)
latest_de  = system(latest_de_cmd)
latest_rms = system(latest_rms_cmd)
latest_E   = system(latest_E_cmd)
nsteps_str = system(nstep_cmd)
latest_dt  = (strlen(timefile) > 0) ? system(latest_dt_cmd) : "NA"
mean_dt    = (strlen(timefile) > 0) ? system(mean_dt_cmd) : "NA"
nsteps     = int(nsteps_str)

# fixed-width window until win is reached, then sliding last-win window
xmax = (nsteps > win) ? nsteps : win
xmin = (nsteps > win) ? (nsteps - win + 1) : 1

# -------- panel 1: log10(|dE|) + Energy --------
unset label
unset xrange
unset y2tics
set xrange [xmin:xmax]
set autoscale y
set autoscale y2

set label 1 sprintf("file: %s", fname)          at screen 0.02,0.98 left
set label 2 sprintf("latest E: %s", latest_E)   at screen 0.40,0.98 left textcolor rgb "cyan"
set label 3 sprintf("latest dE: %s", latest_de) at screen 0.72,0.98 left textcolor rgb "red"
set label 4 sprintf("DAV time: latest %s, mean %s", latest_dt, mean_dt) at screen 0.40,0.945 left textcolor rgb "green"

set xlabel ""
set ylabel "log10(|dE / eV|)" textcolor rgb "red"
set ytics textcolor rgb "red"

set y2label "Energy (eV)" textcolor rgb "cyan"
set y2tics textcolor rgb "cyan"

set grid
set key off

plot \
    "< awk '/^DAV:/{c++; v=$4+0; if (v<0) v=-v; if (v>0) print c, log(v)/log(10)}' ".file using 1:2 with lines lc rgb "red" title "log10(|dE|)", \
    "< awk '/^DAV:/{c++; print c,$3}' ".file using 1:2 axes x1y2 with lines lc rgb "cyan" title "E"

# -------- panel 2: log10(rms) --------
unset label
unset xrange
unset y2tics
set xrange [xmin:xmax]
set autoscale y

set xlabel "DAV iteration"
set ylabel "log10(rms)" textcolor rgb "magenta"
set ytics textcolor rgb "magenta"

set grid
set key off

set label 1 sprintf("latest rms: %s", latest_rms) at graph 0.98,0.95 right textcolor rgb "magenta"

plot \
    "< awk '/^DAV:/{c++; v=$7+0; if (v>0) print c, log(v)/log(10)}' ".file using 1:2 with lines lc rgb "magenta" title "log10(rms)"

# -------- panel 3: observed DAV seconds --------
unset label
unset xrange
unset y2tics
set xrange [xmin:xmax]
set autoscale y

set xlabel "DAV iteration"
set ylabel "seconds / DAV" textcolor rgb "green"
set ytics textcolor rgb "green"

set grid
set key off

set label 1 sprintf("live timing excludes initialization; batch refreshes are averaged per new DAV") at graph 0.02,0.95 left textcolor rgb "green"

plot \
    "< awk 'NF>=4 && $1 !~ /^#/{n++; print $1,$3} END{if(n==0) print 1,0}' ".timefile using 1:2 with linespoints lc rgb "green" title "seconds/DAV"

unset multiplot
