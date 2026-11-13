#!/bin/sh
set -eu

secret_file=/run/secrets/qdrant_api_key
if [ ! -r "$secret_file" ]; then
    echo "Qdrant API-key secret is unavailable" >&2
    exit 1
fi

# Reject NUL bytes before loading the value into a shell variable.
bytes=" $(od -An -v -t x1 "$secret_file" | tr '\n' ' ') "
case "$bytes" in
    *" 00 "*)
    echo "Qdrant API-key secret has an invalid format" >&2
    exit 1
    ;;
esac
unset bytes

key=$(cat "$secret_file")
cr=$(printf '\r')
lf=$(printf '\n_')
lf=${lf%_}
while [ -n "$key" ] && [ "${key#${key%?}}" = "$cr" ]; do
    key=${key%?}
done
case "$key" in
    ''|*"$lf"*|*"$cr"*)
        echo "Qdrant API-key secret has an invalid format" >&2
        exit 1
        ;;
esac

export QDRANT__SERVICE__API_KEY="$key"
unset key
exec ./entrypoint.sh
