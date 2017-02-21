

## Use something a for loop like this to generate and run the various Hyperparameters ##
for CAPABILITY in $COMPUTE_CAPABILITIES; do
  if [[ ! "$CAPABILITY" =~ [0-9]+.[0-9]+ ]]; then
    echo "Invalid compute capability: " $CAPABILITY
    ALL_VALID=0
    break
  fi
done

## Use the above to generate commandline-arguments like so ##
