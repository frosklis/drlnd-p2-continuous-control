tail -n 100 score | awk '{ total += $2 } END { print total/NR }'
tail -n 20 score

