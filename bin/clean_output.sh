#remove all characters except the wave function table from the RCN code ooutput out36
egrep '^[ \.0-9\-]+$' out36  | tail -n +2 | head -n -1 > out36_clean

