score="mlruns_aws/1/16129b2e04e0436394546af9a3eee6ce/metrics/score"
score="mlruns_aws/1/1aca5f4fca804e9eaaa5f8f4588e1dbc/metrics/score"
score="mlruns_aws/1/79a632a83ad44b64958ef0f922b673ad/metrics/score"
score="mlruns_aws/1/cce8880efe034a11819726b56865bf09/metrics/score"
score="mlruns_aws/1/cce8880efe034a11819726b56865bf09/metrics/score"

echo `tail -n 20 $score | awk '{ total += $2 } END { print total/NR }'`  `tail -n 100 $score | awk '{ total += $2 } END { print total/NR }'` `tail -n 200 $score | awk '{ total += $2 } END { print total/NR }'`
tail -n 20 $score
