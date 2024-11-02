source miriam_env/bin/activate
source .env
ulimit -n 100000

# this must be run from the root of the project
if [ ! -f "setup.sh" ]; then
    echo "Please run this script from the root of the project"
    exit 1
fi
export PYTHONPATH=$PYTHONPATH:$(pwd)
