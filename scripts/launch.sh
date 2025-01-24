#!/bin/bash

# Exit on error
set -e

# Export burn_fee so it's available to subprocesses
export burn_fee=2.5

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Always work from the script directory
cd "$SCRIPT_DIR"

# Get the project root directory
PROJECT_ROOT="$(cd .. && pwd)"

# Source paths relative to script location
source_miner="$PROJECT_ROOT/src/synthia/miner/template_miner.py"
source_validator="$PROJECT_ROOT/src/synthia/validator/text_validator.py"

# Error handling
trap 'error_handler $? $LINENO $BASH_LINENO "$BASH_COMMAND" "$(printf "::%s" "${FUNCNAME[@]:-}")"' ERR

# Error handler function
error_handler() {
    local exit_code="$1"
    local line_no="$2"
    local bash_lineno="$3"
    local last_command="$4"
    local func_trace="$5"

    echo "Error occurred in script at line: $line_no"
    if [ -n "$bash_lineno" ]; then
        echo "Function call trace (line numbers): $bash_lineno"
    fi
    if [ -n "$func_trace" ]; then
        echo "Function call trace: $func_trace"
    fi
    echo "Command: $last_command"
    echo "Exit code: $exit_code"
    
    # Cleanup any running processes
    cleanup
    
    exit "$exit_code"
}

# Cleanup function
cleanup() {
    # Kill any running pm2 processes if they exist
    if command -v pm2 &> /dev/null; then
        pm2 delete all &> /dev/null || true
    fi
    
    # Return to original directory if we changed it
    if [ -n "$ORIGINAL_DIR" ]; then
        cd "$ORIGINAL_DIR" || exit
    fi
}

# Store original directory
ORIGINAL_DIR="$PWD"

# Store miner ports file location
MINER_PORTS_FILE="$HOME/.commune/miner_ports.txt"

# Check required commands
check_requirements() {
    local missing_requirements=0
    
    # Check for required commands
    for cmd in python3 pip3 curl; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "Error: $cmd is required but not installed."
            missing_requirements=1
        fi
    done
    
    if [ $missing_requirements -eq 1 ]; then
        echo "Please install missing requirements and try again."
        exit 1
    fi
}

# Function to create miner/validator files
create_module_files() {
    local module_type=$1  # "miner" or "validator"
    local filename=$2
    local classname=$3
    
    echo "Creating $module_type module files..."
    
    local source_file="$PROJECT_ROOT/src/synthia/$module_type/$filename.py"
    if [ ! -f "$source_file" ]; then
        local template_file
        if [ "$module_type" = "miner" ]; then
            template_file="$source_miner"
        else
            template_file="$source_validator"
        fi
        
        # Create directory if it doesn't exist
        mkdir -p "$PROJECT_ROOT/src/synthia/$module_type"
        
        # Copy template and replace class name
        cp "$template_file" "$source_file"
        sed -i "s/${module_type^}_1/$classname/g" "$source_file"
        echo "$module_type module created at $source_file"
    fi
}

# Install Synthia
install_synthia() {
    echo "Installing Synthia"
    
    # Move to the root directory of the project if we're in scripts
    if [[ "$PWD" == */scripts ]]; then
        cd ..
    fi

    if [ ! -x "/usr/bin/python3" ]; then
        echo "Python 3 is not installed. Please install Python 3 and try again."
        exit 1
    fi
    if [ ! -x "/usr/bin/pip3" ]; then
        echo "Python 3 is not installed. Please install Python 3 and try again."
        exit 1
    fi

    # Setting up virtual environment
    python3 -m venv .venv
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
    python3 -m pip install --upgrade pip
    pip3 install setuptools wheel gnureadline

    # Installing dependencies
    pip3 install -r requirements.txt

    # Installing synthia
    pip3 install -e .

    # Installing communex
    pip3 install --upgrade communex
    echo "Synthia installed."
}

# Sets up the environment for the miner or validator
setup_environment() {
    echo "Setting up environment..."
    
    # Ensure we have the required system packages for Python
    if command -v apt-get &> /dev/null; then
        echo "Installing required system packages..."
        sudo apt-get update
        sudo apt-get install -y python3-venv python3-full
    else
        echo "This script requires apt-get package manager (Ubuntu/Debian)"
        exit 1
    fi
    
    # Create project virtual environment if it doesn't exist
    if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$PROJECT_ROOT/.venv"
    fi
    
    # Activate the virtual environment
    echo "Activating virtual environment..."
    VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
    if [ ! -f "$VENV_ACTIVATE" ]; then
        echo "Error: Virtual environment activation script not found at $VENV_ACTIVATE"
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$VENV_ACTIVATE"
    
    # Install or upgrade pip in the virtual environment
    python3 -m pip install --upgrade pip
    
    # Install Poetry for dependency management if not installed
    if ! command -v poetry &> /dev/null; then
        echo "Installing Poetry for dependency management..."
        python3 -m pip install poetry
    fi
    
    # Install project dependencies using Poetry
    echo "Installing project dependencies..."
    cd "$PROJECT_ROOT"
    poetry config virtualenvs.create false  # Don't create Poetry venv, use our own
    poetry install
    
    # Create commune directories if they don't exist
    echo "Creating commune directories..."
    mkdir -p "$HOME/.commune/key"
    mkdir -p "$HOME/.commune/db"
    
    # Create config.env from sample if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/env/config.env" ] && [ -f "$PROJECT_ROOT/env/sample.config.env" ]; then
        echo "Creating initial config.env from sample..."
        cp "$PROJECT_ROOT/env/sample.config.env" "$PROJECT_ROOT/env/config.env"
        echo "Please edit env/config.env with your settings before starting the miner."
    fi
    
    echo "Installation complete! Your environment is ready."
    echo ""
    echo "Important: Edit env/config.env with your settings:"
    echo "- API key for your inference endpoint"
    echo "- Base URL for your inference endpoint"
    echo "- Model identifier (if required)"
    echo "- Any additional configuration needed for your setup"
    echo ""
    
    # Activate the virtual environment and start menu
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
    cd "$PROJECT_ROOT"
    
    # Start the menu directly
    if [ -z "$SYNTHIA_SHELL" ]; then
        export SYNTHIA_SHELL=1
        export PS1="(synthia) $PS1"
        main_menu
    fi
}

# Function to ensure we're in the virtual environment
ensure_venv() {
    # Skip if we're in setup mode or already in virtual environment
    if [ "$1" = "--setup" ] || [ -n "$VIRTUAL_ENV" ]; then
        return
    fi

    if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        echo "Virtual environment not found. Running setup..."
        "$0" --setup
        return
    fi

    # If we're not in the virtual environment, activate it
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Activating virtual environment..."
        VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
        if [ ! -f "$VENV_ACTIVATE" ]; then
            echo "Error: Virtual environment activation script not found at $VENV_ACTIVATE"
            exit 1
        fi
        # shellcheck source=/dev/null
        source "$VENV_ACTIVATE"
        cd "$PROJECT_ROOT"
        
        # Set up shell environment
        if [ -z "$SYNTHIA_SHELL" ]; then
            export SYNTHIA_SHELL=1
            export PS1="(synthia) $PS1"
            main_menu
            exit 0
        fi
    fi
}

# Function to configure the module launch
configure_launch() {
    # Initialize is_update flag
    is_update="false"
    
    # Enter the path of the module
    echo "The module name should be in the format of \"Namespace.Miner_X\" (eg. Rabbit.Miner_0)"
    while true; do
        read -r -p "Module name: " key_name

        # Check if the module path is valid
        if [ -z "$key_name" ] || [[ ! "$key_name" =~ ^[A-Za-z0-9]+\.[A-Za-z0-9_]+$ ]]; then
            echo "Error: Must provide a valid module name in the format Namespace.Miner_X"
            continue
        fi
        break
    done

    # Check if the key exists by trying to read it
    if [ ! -f "$HOME/.commune/key/$key_name.json" ]; then
        echo "Key '$key_name' does not exist."
        read -r -p "Would you like to create it? [y/N] " create_key_response
        if [[ "$create_key_response" =~ ^[Yy]$ ]]; then
            if ! create_key "$key_name"; then
                echo "Failed to create key. Exiting..."
                exit 1
            fi
            echo "Key created successfully."
        else
            echo "Cannot proceed without a valid key. Exiting..."
            exit 1
        fi
    fi

    # Check current balance before proceeding
    local free_balance
    if ! free_balance=$(comx balance free-balance "$key_name" 2>/dev/null | tail -n 1 | grep -oP '^[\d.]+' || echo "0"); then
        echo "Error: Failed to get wallet balance."
        exit 1
    fi

    if [ -z "$free_balance" ] || [ "$free_balance" = "0" ]; then
        echo "Warning: No balance found or balance is 0 COMAI"
        echo "You will need to transfer some COMAI to this wallet before registering."
        read -r -p "Would you like to transfer balance now? [y/N] " transfer_now
        if [[ "$transfer_now" =~ ^[Yy]$ ]]; then
            transfer_balance
            # Recheck balance after transfer
            if ! free_balance=$(comx balance free-balance "$key_name" 2>/dev/null | tail -n 1 | grep -oP '^[\d.]+' || echo "0"); then
                echo "Error: Failed to get updated wallet balance."
                exit 1
            fi
            if [ -z "$free_balance" ] || [ "$free_balance" = "0" ]; then
                echo "Still no balance after transfer. Cannot proceed."
                exit 1
            fi
        else
            echo "Cannot proceed without balance. Exiting..."
            exit 1
        fi
    fi

    # Enter the IP and port of the module
    while true; do
        read -r -p "Module IP address for registration (the IP other nodes will use to connect to this miner): " registration_host
        if [ -z "$registration_host" ]; then
            echo "You must provide an IP address that other nodes can use to connect to this miner"
            continue
        fi
        if validate_ip "$registration_host"; then
            break
        fi
        echo "Please enter a valid IP address"
    done
    # Get configured port range
    read -r start_port end_port <<< "$(get_port_range)"
    
    # Check if port is already assigned to this miner
    local saved_port=""
    if [ -f "$MINER_PORTS_FILE" ]; then
        saved_port=$(grep "^$key_name:" "$MINER_PORTS_FILE" | cut -d':' -f2)
    fi

    if [ -n "$saved_port" ]; then
        # Verify saved port is still in valid range
        if [ "$saved_port" -ge "$start_port" ] && [ "$saved_port" -le "$end_port" ]; then
            echo "Found previously registered port for $key_name: $saved_port"
            port=$saved_port
        else
            echo "Warning: Previously saved port $saved_port is outside configured range ($start_port-$end_port)"
            saved_port=""
        fi
    fi

    if [ -z "$saved_port" ]; then
        while true; do
            # Find the next available port in the configured range
            local suggested_port=$start_port
            if [ -f "$MINER_PORTS_FILE" ]; then
                while [ "$suggested_port" -le "$end_port" ] && grep -q ":$suggested_port$" "$MINER_PORTS_FILE"; do
                    suggested_port=$((suggested_port + 1))
                done
            fi
            
            if [ "$suggested_port" -gt "$end_port" ]; then
                echo "No available ports in range $start_port-$end_port"
                echo "Would you like to:"
                echo "1. Configure a different port range"
                echo "2. Enter a specific port"
                echo "3. Exit"
                read -r -p "Choose an option (1-3): " port_option
                case "$port_option" in
                    1)
                        configure_port_range
                        read -r start_port end_port <<< "$(get_port_range)"
                        continue
                        ;;
                    2)
                        suggested_port=""
                        ;;
                    *)
                        echo "Exiting..."
                        exit 1
                        ;;
                esac
            fi
            
            if [ -n "$suggested_port" ]; then
                echo "Suggested available port: $suggested_port (from range $start_port-$end_port)"
            else
                echo "Enter a port number between $start_port and $end_port"
            fi
            
            read -r -p "Module port (press Enter to use suggested port): " port
            [ -z "$port" ] && port=$suggested_port
            
            if ! validate_port "$port"; then
                continue
            fi
            
            # Verify port is in configured range
            if [ "$port" -lt "$start_port" ] || [ "$port" -gt "$end_port" ]; then
                echo "Port must be between $start_port and $end_port"
                continue
            fi
            
            # Check if port is already in use by another miner
            if [ -f "$MINER_PORTS_FILE" ] && grep -q ":$port$" "$MINER_PORTS_FILE"; then
                echo "Port $port is already assigned to another miner"
                continue
            fi
            break
        done
        
        # Save the port assignment
        mkdir -p "$(dirname "$MINER_PORTS_FILE")"
        echo "$key_name:$port" >> "$MINER_PORTS_FILE"
    fi

    # Enter the netuid of the module with validation
    while true; do
        read -r -p "Deploying to subnet (default 3): " netuid
        [ -z "$netuid" ] && netuid=3
        validate_number "$netuid" 0 100 && break
        echo "Please enter a valid subnet number (0-100)"
    done

    # Determine if module needs staking based on module type
    if [[ "$key_name" == *".Validator"* ]]; then
        needs_stake="true"
    elif [[ "$key_name" == *".Miner"* ]]; then
        needs_stake="true"
    else
        needs_stake="false"
    fi

    # Select if a balance needs to be transfered to the key
    echo "Transfer staking balance to the module key."
    echo "You can skip this step if you have enough balance on your key."
    echo "The sending key must be in the ~/.commune/key folder with enough com to transfer."
    read -r -p "Transfer balance (y/n): " transfer_balance
    if [ "$transfer_balance" = "y" ]; then
        transfer_balance
    fi
    echo ""

    # Check if the module needs to be staked
    if [ "$needs_stake" = "true" ]; then
        echo "Set the stake. This is the amount of tokens that will be staked by the module."
        echo "Validators require a balance of 5200, not including fees, to vote."
        echo "Miners require a balance of 256, not including fees, to mine."
        echo "There will be a burn fee that starts at 10 com and scales based on demand"
        echo "will be burned as a fee to stake. Make sure you have enough to cover the cost."
        read -r -p "Set stake: " stake
        echo "Setting stake: $stake"
        echo ""
    fi

    # Enter the delegation fee
    if [ "$is_update" = "true" ]; then
        echo "Set the delegation fee. This the percentage of the emission that are collected as a fee to delegate the staked votes to the module."
        read -r -p "Delegation fee (default 20) int: " delegation_fee
        echo ""
    fi

    # Check it is above minimum
    if [ "$delegation_fee" -lt 5 ] || [ "$delegation_fee" = "" ]; then
        echo "Minimum delegation fee is 5%. Setting to 5%"
        delegation_fee=5
        echo "Module delegation fee: $delegation_fee"
        echo ""
    fi

    # Enter the metadata
    if [ "$is_update" = "true" ]; then
        echo "Set the metadata. This is an optional field."
        echo "It is a JSON object that is passed to the module in the format:"
        echo "{\"key\": \"value\"}."
        read -r -p "Add metadata (y/n): " choose_metadata
        if [ "$choose_metadata" = "y" ]; then
            read -r -p "Enter metadata object: " metadata
            echo "Module metadata: $metadata"
        fi
        echo ""
    fi

    # Confirm settings
    echo "Confirm module settings:"
    echo "Module path:        $module_path"
    echo "Module IP address for registration:  $registration_host"
    echo "Module port:        $port"
    echo "Module netuid:      $netuid"
    echo "Module key name:    $key_name"
    if [ "$needs_stake" = "true" ]; then
        echo "Module stake:       $stake"
    fi
    if [ "$is_update" = "true" ]; then
        echo "Delegation fee:     $delegation_fee"
        echo "Metadata:           $metadata"
    fi
    read -r -p "Confirm settings (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        echo "Deploying..."
        echo ""
    else
        echo "Aborting..."
        exit 1
    fi

    # Export the variables for use in the bash script
    export MODULE_PATH="$module_path"
    export MODULE_IP="0.0.0.0"  # Always use 0.0.0.0 for serving
    export MODULE_REGISTRATION_IP="$registration_host"  # Use real IP for registration
    export MODULE_PORT="$port"
    export MODULE_NETUID="$netuid"
    export MODULE_KEYNAME="$key_name"
    export MODULE_STAKE="$stake"
    export MODULE_DELEGATION_FEE="$delegation_fee"
    export MODULE_METADATA="$metadata"
}

# Function to create a key
create_key() {
    echo "Creating key"
    echo "This creates a json key in ~/.commune/key with the given name."
    echo "Once you create the key you will want to save the mnemonic somewhere safe."
    echo "The mnemonic is the only way to recover your key if it lost then the key is unrecoverable."
    echo "Note that commune does not encrypt the key file so do not fund a key on an unsafe machine."

    if [ -z "$key_name" ]; then
        read -r -p "Key name: " key_name
    fi
    comx key create "$key_name"
    echo "This is your key. Save the mnemonic somewhere safe."
    cat ~/.commune/key/"$key_name".json
    echo "$key_name created and saved at ~/.commune/key/$key_name.json"
}

# Function to perform a balance transfer
transfer_balance() {
    echo "Initiating Balance Transfer"
    echo "There is a 2.5 com fee on the balance of the transfer."
    echo "Example: 300 com transfered will arrive as 297.5 com"
    read -r -p "From Key (sender): " key_from
    read -r -p "Amount to Transfer: " amount
    if [ -z "$key_name" ]; then
        read -r -p "To Key (recipient): " key_to
    else
        key_to="$key_name"
    fi
    comx balance transfer "$key_from" "$amount" "$key_to"
    echo "Transfer of $amount from $key_from to $key_to initiated."
}

# Function to unstake balance from a module
unstake_and_transfer_balance() {
    local key_from="${1:-}"
    local key_to="${2:-}"
    local key_to_transfer="${3:-}"
    local subnet="${4:-}"
    local amount="${5:-}"

    if [ -z "$key_from" ] || [ -z "$key_to" ] || [ -z "$key_to_transfer" ] || [ -z "$subnet" ] || [ -z "$amount" ]; then
        echo "Initiating Balance Unstake"
        read -r -p "Unstake from: " key_from
        read -r -p "Unstake to: " key_to
        read -r -p "Transfer to: " key_to_transfer
        read -r -p "Amount to unstake: " amount
    fi

    amount_minus_half=$(echo "$amount - 0.5" | awk '{print $1 - 0.5}')
    comx balance unstake "$key_from" "$amount" "$key_to"
    echo "$amount COM unstaked from $key_from to $key_to"

    echo "Initiating Balance Transfer"
    comx balance transfer "$key_to" "$amount_minus_half" "$key_to_transfer"
    echo "Transfer of $amount_minus_half from $key_to to $key_to_transfer initiated."
}

# Function to unstake and transfer balance of all modules
unstake_and_transfer_balance_all() {
  echo "Unstaking and transferring balance of all modules..."

  # Get the module names of all modules in the .commune/key directory
  modulenames=$(find "$HOME/.commune/key" -type f -name "*_*" -print0 | 
    xargs -0 basename -a | 
    sed 's/\.[^.]*$//' | 
    tr '\n' ' ')

  # Store the module names in an array
  IFS=' ' read -r -a modulenames_array <<< "$modulenames"

  unstake_and_transfer_balance_multiple "${modulenames_array[@]}"
}

# Function to unstake and transfer balance of all modules
unstake_and_transfer_balance_name() {

  declare -a module_names=()

  echo "Enter module names ('.' to stop entering module names):"
  while true; do
      read -r -p "Module name: " module_name
      if [[ $module_name == "." ]]; then
          break
      fi
      module_names+=("$module_name")
  done

  # Get the module names of all modules in the .commune/key directory that match the provided module names
  modulenames=$(find "$HOME/.commune/key" -type f -name "*_*" -print0 | 
    xargs -0 basename -a | 
    sed 's/\.[^.]*$//' | 
    grep -E "$(IFS="|"; echo "${module_names[*]}")" | 
    tr '\n' ' ')  # Store the module names in an array
  IFS=' ' read -r -a modulenames_array <<< "$modulenames"

  unstake_and_transfer_balance_multiple "${modulenames_array[@]}"
}

# Function to unstake and transfer balance of multiple modules
unstake_and_transfer_balance_multiple() {
    declare -a module_names=()

    # Check if any module names are passed as arguments
    if [[ $# -gt 0 ]]; then
        module_names=("$@")
    else
        echo "Enter module names ('.' to stop entering module names):"
        while true; do
            read -r -p "Module name: " module_name
            if [[ $module_name == "." ]]; then
                break
            fi
            module_names+=("$module_name")
        done
    fi

    # Ask the user for the amount
    read -r -p "Amount to unstake from each miner: " amount

    # Ask the user for the key to transfer the balance to
    read -r -p "Key to transfer balance to: " key_to_transfer

    # Now the module_names array contains the names of the modules entered by the user
    echo "Module names entered: ${module_names[*]@Q}"

    # Now the amounts array contains the amounts entered by the user
    echo "Amount to unstake and transfer: $amount"

    # You can now use the module_names and amounts arrays to perform the unstake and transfer balance operations for each module
    for module_name in "${module_names[@]}"; do
        echo "Processing module: $module_name"
        unstake_and_transfer_balance "$module_name" "$module_name" "$key_to_transfer" "$subnet" "$amount"
    done

    # Print the total amount of balance transferred - amount * number of modules
    echo "Successfully transferred: $(echo "$amount * ${#module_names[@]}" | bc -l) to $key_to_transfer"
}

# Function to transfer and stake balance of multiple modules from one key
transfer_and_stake_multiple() {
    declare -a module_names=()

    # Ask the user for the amount
    read -r -p "Amount to stake to each miner: " amount

    echo "Enter module names ('.' to stop entering module names):"
    while true; do
        read -r -p "Module name: " module_name
        if [[ $module_name == "." ]]; then
            break
        fi
        module_names+=("$module_name")
    done


    # Ask the user for the key to transfer the balance to
    read -r -p "Key to transfer balance from: " key_from


    # transfer balance and stake to each miner
    for i in "${!module_names[@]}"; do
        key_to="${module_names[i]}"
    echo "Initiating Balance Transfer"
    comx balance transfer "$key_from" "$amount" "$key_to"
    echo "Transfer of $amount from $key_from to $key_to completed."
    amount_minus_half=$(echo "$amount - 0.5" | awk '{print $1 - 0.5}')
    comx balance stake "$key_to" "$amount_minus_half" "$key_to"
    echo "$amount_minus_half COM staked from $key_to to $key_to"
        
    done
}

# Function to serve a miner
serve_miner() {
    echo "Serving Miner"
    
    # Get module name input
    echo "The module name should be in the format of \"Namespace.Miner_X\" (eg. Rabbit.Miner_0)"
    while true; do
        read -r -p "Module name: " key_name

        # Check if the module path is valid
        if [ -z "$key_name" ] || [[ ! "$key_name" =~ ^[A-Za-z0-9]+\.[A-Za-z0-9_]+$ ]]; then
            echo "Error: Must provide a valid module name in the format Namespace.Miner_X"
            continue
        fi
        break
    done

    # Extract the namespace and class name
    local namespace="${key_name%%.*}"
    local classname="${key_name#*.}"
    local module_path="synthia.miner.${namespace}.${classname}"
    
    # Move to the root directory if we're in scripts
    if [[ "$PWD" == */scripts ]]; then
        cd ..
    fi
    
    # Ensure we're in a virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Activating virtual environment..."
        VENV_ACTIVATE="$ORIGINAL_DIR/.venv/bin/activate"
        if [ ! -f "$VENV_ACTIVATE" ]; then
            echo "Error: Virtual environment activation script not found at $VENV_ACTIVATE"
            exit 1
        fi
        # shellcheck source=/dev/null
        source "$VENV_ACTIVATE"
    fi
    
    # Install the package in editable mode if not already installed
    if ! pip show synthia &> /dev/null; then
        echo "Installing Synthia package..."
        pip install -e .
    fi
    
    # Clean up any existing PM2 processes with this name
    if command -v pm2 &> /dev/null; then
        pm2 delete "$key_name" &> /dev/null || true
    fi
    
    # Set minimal environment variables needed for serving
    export MODULE_PATH="$module_path"
    export MODULE_IP="0.0.0.0"  # Always use 0.0.0.0 for serving
    export MODULE_KEYNAME="$key_name"
    
    # Get the port from miner_ports.txt
    MINER_PORTS_FILE="$HOME/.commune/miner_ports.txt"
    if [ ! -f "$MINER_PORTS_FILE" ]; then
        echo "Error: No miner ports file found at $MINER_PORTS_FILE"
        echo "Please register the miner first using option 5"
        return 1
    fi
    
    local port
    port=$(grep "^$key_name:" "$MINER_PORTS_FILE" | cut -d':' -f2)
    
    if [ -z "$port" ]; then
        echo "Error: No port found for miner $key_name"
        echo "Please register the miner first using option 5"
        return 1
    fi
    
    # Create the command to run the miner using ModuleServer with proper key loading
    local run_cmd="from synthia.miner.${namespace} import ${classname}; from communex.module.server import ModuleServer; from communex.compat.key import classic_load_key; from communex.module._rate_limiters.limiters import StakeLimiterParams; import uvicorn; keypair = classic_load_key('${key_name}'); module = ${classname}(); stake_limiter = StakeLimiterParams(epoch=800, cache_age=600); server = ModuleServer(module, keypair, subnets_whitelist=[3], limiter=stake_limiter); app = server.get_fastapi_app(); app.include_router(module.router); uvicorn.run(app, host='0.0.0.0', port=${port})"
    
    # Start the miner with PM2
    echo "Starting miner with PM2..."
    if ! pm2 start --name "$key_name" python3 -- -c "import sys; sys.path.append('.'); $run_cmd"; then
        echo "Error: Failed to start miner with PM2"
        return 1
    fi
    
    echo "Miner served. View logs with: pm2 logs $key_name"
    
    # Return status for the calling function
    return 0
}

# Function to deploy a miner
deploy_miner() {
    echo "Deploying Miner"
    configure_launch
    register_miner
    serve_miner "$key_name"
}

# Function to serve a validator
serve_validator() {
    echo "Serving Validator"
    
    # Check for required environment variables
    if [ -z "$ANTHROPIC_api_key" ]; then
        echo "Error: ANTHROPIC_api_key environment variable is not set"
        echo "Please set it in your env/config.env file or export it directly"
        return 1
    fi
    
    # Move to the root directory if we're in scripts
    if [[ "$PWD" == */scripts ]]; then
        cd ..
    fi
    
    # Ensure we're in a virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Activating virtual environment..."
        VENV_ACTIVATE="$ORIGINAL_DIR/.venv/bin/activate"
        if [ ! -f "$VENV_ACTIVATE" ]; then
            echo "Error: Virtual environment activation script not found at $VENV_ACTIVATE"
            exit 1
        fi
        # shellcheck source=/dev/null
        source "$VENV_ACTIVATE"
    fi
    
    # Extract the namespace and class name
    local namespace="${key_name%%.*}"
    local classname="${key_name#*.}"
    local module_path="synthia.validator.${namespace}.${classname}"
    
    # Clean up any existing PM2 processes with this name
    echo "Cleaning up any existing validator processes..."
    pm2 delete "$module_path" 2>/dev/null || true
    
    # Start the validator with pm2, passing all required arguments
    echo "Starting validator..."
    if pm2 start --name "$module_path" \
        --interpreter python3 \
        -f ./src/synthia/cli.py -- \
        --key_name "$key_name" \
        --host "$host" \
        --port "$port" \
        validator "$filename"; then
        
        echo "Validator started successfully."
        echo "Use 'pm2 logs' to view validator output"
        echo "Use 'pm2 stop $module_path' to stop the validator"
        return 0
    else
        echo "Failed to start validator. Check the logs for more information."
        return 1
    fi
}

# Function to register a miner
register_miner() {
    echo "Registering Miner"
    
    # Extract the namespace part (before the dot) and create the miner file if it doesn't exist
    local namespace="${key_name%%.*}"
    local classname="${key_name#*.}"
    
    # Validate namespace and classname
    if [ -z "$namespace" ] || [ -z "$classname" ] || [ "$namespace" = "$key_name" ]; then
        echo "Error: Invalid module name. Must be in format Namespace.Miner_X"
        return 1
    fi
    
    # Check if key is already registered on the network
    if comx module info "$key_name" --netuid "$netuid" &>/dev/null; then
        echo "Key '$key_name' is already registered on subnet $netuid"
        echo "Proceeding with existing registration..."
        return 0
    fi

    local miner_file="$PROJECT_ROOT/src/synthia/miner/${namespace}.py"
    local init_file="$PROJECT_ROOT/src/synthia/miner/__init__.py"
    
    if [ ! -f "$miner_file" ]; then
        echo "Creating new miner file: $miner_file"
        mkdir -p "$PROJECT_ROOT/src/synthia/miner"
        cp "$source_miner" "$miner_file"
        
        # Replace the class name in the template
        sed -i "s/Miner_1/$classname/g" "$miner_file"
        
        # Add import to __init__.py if not already there
        if [ ! -f "$init_file" ]; then
            echo "Creating __init__.py..."
            echo "from . import $namespace" > "$init_file"
            echo "" >> "$init_file"
            echo "__all__ = ['$namespace']" >> "$init_file"
        elif ! grep -q "from . import $namespace" "$init_file"; then
            # Add the import at the top of the file
            sed -i "1i from . import $namespace" "$init_file"
            # Update __all__ list
            if grep -q "__all__" "$init_file"; then
                # Add to existing __all__ list if namespace not already there
                if ! grep -q "'$namespace'" "$init_file"; then
                    sed -i "s/__all__ = \[/__all__ = \['$namespace', /" "$init_file"
                fi
            else
                # Create new __all__ list
                echo "" >> "$init_file"
                echo "__all__ = ['$namespace']" >> "$init_file"
            fi
        fi
    fi

    # The module path should point to the specific class
    local module_path="synthia.miner.${namespace}.${classname}"
    
    # First register the miner
    echo "Registering miner with network..."
    if [ -z "$MODULE_REGISTRATION_IP" ]; then
        MODULE_REGISTRATION_IP="$registration_host"
    fi
    
    # Register with key_name directly as the name, since that's what we want to use for identification
    if ! comx module register --ip "$MODULE_REGISTRATION_IP" --port "$port" "$key_name" "$key_name" "$netuid"; then
        echo "Error: Failed to register miner"
        return 1
    fi
    
    # Get current balance for staking
    local free_balance
    free_balance=$(comx balance free-balance "$key_name" 2>/dev/null | tail -n 1 | grep -oP '^[\d.]+' || echo "0")
    local max_stake
    max_stake=$(echo "$free_balance - 1" | bc -l)
    echo "Available balance: $free_balance COMAI (maximum stakeable amount: $max_stake COMAI)"
    
    # Ask if user wants to stake
    read -r -p "Would you like to stake tokens? [y/N] " stake_response
    if [[ "$stake_response" =~ ^[Yy]$ ]]; then
        while true; do
            read -r -p "Enter amount to stake (max $max_stake COMAI): " stake_amount
            if [[ "$stake_amount" =~ ^[0-9]+\.?[0-9]*$ ]] && \
               [ "$(echo "$stake_amount <= $max_stake" | bc -l)" -eq 1 ] && \
               [ "$(echo "$stake_amount > 0" | bc -l)" -eq 1 ]; then
                echo "Staking $stake_amount COMAI to miner..."
                if ! comx balance stake "$key_name" "$stake_amount" "$key_name"; then
                    echo "Error: Failed to stake tokens"
                    return 1
                fi
                break
            else
                echo "Invalid amount. Please enter a number between 0 and $max_stake"
            fi
        done
    fi
    
    echo "Miner registered successfully."
    return 0
}

# Function to register a validator
register_validator() {
    echo "Registering Validator"
    
    # Extract the namespace part (before the dot) and create proper module path
    local namespace="${key_name%%.*}"
    local classname="${key_name#*.}"
    local module_path="synthia.validator.${namespace}.${classname}"
    
    # First register the validator
    echo "Registering validator with network..."
    comx module register --ip "$MODULE_REGISTRATION_IP" --port "$port" "$module_path" "$key_name" "$netuid"
    
    if [ -n "$metadata" ]; then
        comx module update "$module_path" "$key_name" --metadata "$metadata"
    fi
    
    # Check current balance
    echo "Checking current balance..."
    local free_balance
    free_balance=$(comx balance free-balance "$key_name" 2>/dev/null | tail -n 1 | grep -oP '^[\d.]+' || echo "0")
    local max_stake
    max_stake=$(echo "$free_balance - 1" | bc -l)
    echo "Available balance: $free_balance COMAI (maximum stakeable amount: $max_stake COMAI)"
    
    # Ask if user wants to stake
    read -r -p "Would you like to stake tokens? [y/N] " stake_response
    if [[ "$stake_response" =~ ^[Yy]$ ]]; then
        while true; do
            read -r -p "Enter amount to stake (max $max_stake COMAI): " stake_amount
            if [[ "$stake_amount" =~ ^[0-9]+\.?[0-9]*$ ]] && \
               [ "$(echo "$stake_amount <= $max_stake" | bc -l)" -eq 1 ] && \
               [ "$(echo "$stake_amount > 0" | bc -l)" -eq 1 ]; then
                echo "Staking $stake_amount COMAI to validator..."
                comx balance stake "$key_name" "$stake_amount" "$key_name"
                break
            else
                echo "Invalid amount. Please enter a number between 0 and $max_stake"
            fi
        done
    fi
    
    echo "Validator registered and staked."
}

# Function to update a module
update_module() {
    echo "Updating Module"
    # Usage: comx module update [OPTIONS] KEY NETUID
    local options=""
    [ -n "$host" ] && options="$options --ip $host"
    [ -n "$port" ] && options="$options --port $port"
    [ -n "$module_path" ] && options="$options --name $module_path"
    
    # Execute update command with built options
    comx module update "$options" "$key_name" "$netuid"
    echo "Module updated."
}

# Function to deploy a validator
deploy_validator() {
    echo "Serving Validator"
    serve_validator
    echo "Registering Validator"
    register_validator
    echo "Validator deployed."
}

# Function to get port range
get_port_range() {
    local start_port=10001
    local end_port=10200

    if [ -f "$HOME/.commune/port_config.txt" ]; then
        local temp_start
        local temp_end
        temp_start=$(grep "^START_PORT=" "$HOME/.commune/port_config.txt" | cut -d'=' -f2)
        temp_end=$(grep "^END_PORT=" "$HOME/.commune/port_config.txt" | cut -d'=' -f2)
        if [ -n "$temp_start" ] && [ -n "$temp_end" ]; then
            start_port=$temp_start
            end_port=$temp_end
        fi
    fi

    printf "%d %d" "$start_port" "$end_port"
}

# Function to set port range
configure_port_range() {
    echo "Configure port range for miners"
    echo "These ports must be open and accessible from the network"
    
    local current_range
    current_range=$(get_port_range)
    local current_start
    local current_end
    read -r current_start current_end <<< "$current_range"
    
    while true; do
        read -r -p "Enter start port (current: $current_start): " start_port
        if [ -z "$start_port" ]; then
            start_port=$current_start
            break
        fi
        if validate_port "$start_port"; then
            break
        fi
    done

    while true; do
        read -r -p "Enter end port (current: $current_end): " end_port
        if [ -z "$end_port" ]; then
            end_port=$current_end
            break
        fi
        if validate_port "$end_port" && [ "$end_port" -gt "$start_port" ]; then
            break
        fi
        echo "End port must be greater than start port ($start_port)"
    done

    # Create the .commune directory if it doesn't exist
    if [ ! -d "$HOME/.commune" ]; then
        mkdir -p "$HOME/.commune"
    fi

    # Write the configuration using a temporary file for atomicity
    local temp_file
    temp_file=$(mktemp)
    {
        echo "START_PORT=$start_port"
        echo "END_PORT=$end_port"
    } > "$temp_file"
    mv "$temp_file" "$HOME/.commune/port_config.txt"
    
    echo "Port range configured: $start_port-$end_port"
}

# Function to serve a test miner
serve_test_miner() {
    echo "Starting test miner with higher rate limits..."
    
    # Move to the root directory if we're in scripts
    if [[ $PWD == */scripts ]]; then
        cd ..
    fi

    # Prompt for miner name
    echo "Enter the name of an existing miner (e.g., Rabbit.Miner_0)"
    read -r -p "Miner name: " key_name

    # Check if miner ports file exists
    MINER_PORTS_FILE="$HOME/.commune/miner_ports.txt"
    if [ ! -f "$MINER_PORTS_FILE" ]; then
        echo " No miner ports file found at $MINER_PORTS_FILE"
        return 1
    fi
    
    # Look up the port from saved configuration
    local port
    port=$(grep "^$key_name:" "$MINER_PORTS_FILE" | cut -d':' -f2)
    
    if [ -z "$port" ]; then
        echo " No port found for miner $key_name"
        echo "Make sure the miner is registered and running"
        return 1
    fi

    echo "Found port $port for miner $key_name"
    
    # First stop any existing instance
    pm2 delete "$key_name" 2>/dev/null || true
    
    # Create a shell script to run the miner with environment variables
    local run_script="/tmp/run_miner_$key_name.sh"
    cat > "$run_script" << EOF
#!/bin/bash
source "${ORIGINAL_DIR}/.venv/bin/activate"

# Set higher IP rate limits
export CONFIG_IP_LIMITER_BUCKET_SIZE=1000
export CONFIG_IP_LIMITER_REFILL_RATE=100

# Set higher stake rate limits
export CONFIG_STAKE_LIMITER_EPOCH=10
export CONFIG_STAKE_LIMITER_CACHE_AGE=600
export CONFIG_STAKE_LIMITER_TOKEN_RATIO=100

exec python3 -m synthia.miner.cli "$key_name" --port "$port" --ip "0.0.0.0"
EOF
    chmod +x "$run_script"
    
    echo "Starting miner with rate limits:"
    echo "  IP Limiter: bucket_size=1000, refill_rate=100"
    echo "  Stake Limiter: epoch=10, token_ratio=100"
    
    # Start the miner using the shell script
    pm2 start "$run_script" --name "$key_name" --update-env
    
    echo "Miner started in test mode. Press Enter to continue..."
    read -r
}

# Function to test a miner
test_miner() {
    local miner_name=$1
    
    if [ -z "$miner_name" ]; then
        read -r -p "Enter miner name (e.g., Rabbit.Miner_0): " miner_name
    fi
    
    # Check if miner ports file exists
    MINER_PORTS_FILE="$HOME/.commune/miner_ports.txt"
    if [ ! -f "$MINER_PORTS_FILE" ]; then
        echo " No miner ports file found at $MINER_PORTS_FILE"
        return 1
    fi
    
    # Look up the port from saved configuration
    local port
    port=$(grep "^$miner_name:" "$MINER_PORTS_FILE" | cut -d':' -f2)
    
    if [ -z "$port" ]; then
        echo " No port found for miner $miner_name"
        echo "Make sure the miner is registered first"
        return 1
    fi

    echo "Testing miner $miner_name on port $port..."
    
    # Save current directory
    local current_dir="$PWD"
    
    # Move to scripts directory if we're not already there
    if [[ $PWD != */scripts ]]; then
        cd scripts || return 1
    fi
    
    # Run the test script and capture its output
    echo "Running test..."
    echo "----------------------------------------"
    if python3 test_miner.py "$port" "Test request to verify miner functionality" "$miner_name"; then
        echo "----------------------------------------"
        echo "Test completed successfully!"
    else
        echo "----------------------------------------"
        echo "Test failed!"
    fi
    
    # Return to original directory
    cd "$current_dir" || return 1

    # Ask user what to do next
    while true; do
        echo -e "\nWhat would you like to do?"
        echo "1) Return to main menu"
        echo "2) Run test again"
        echo "3) Exit"
        read -r -p "Choose an option (1-3): " choice
        
        case $choice in
            1) return 0 ;;
            2) test_miner "$miner_name" ;;
            3) exit 0 ;;
            *) echo "Invalid option" ;;
        esac
        
        echo ""
        read -r -p "Press Enter to continue..."
    done
}

# Helper Functions
validate_ip() {
    local ip=$1
    if [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        return 0
    else
        echo "Invalid IP address format. Using default: 0.0.0.0"
        return 1
    fi
}

validate_port() {
    local port=$1
    if [[ $port =~ ^[0-9]+$ ]] && [ "$port" -ge 1 ] && [ "$port" -le 65535 ]; then
        return 0
    else
        echo "Invalid port number. Using default: 10001"
        return 1
    fi
}

validate_number() {
    local num=$1
    local min=$2
    local max=$3
    if [[ $num =~ ^[0-9]+$ ]] && [ "$num" -ge "$min" ] && [ "$num" -le "$max" ]; then
        return 0
    else
        return 1
    fi
}

show_help() {
    cat << EOF
Synthia Deployment Script
Usage: ./launch.sh [OPTIONS] [COMMAND] [PARAMETERS]

Commands:
  serve_miner <name> [--test-mode]     Start a miner process
    Parameters:
      name                             Miner name in format Namespace.Miner_0
      --test-mode                      Use higher rate limits for testing

  serve_validator <name>               Start a validator process
    Parameters:
      name                             Validator name in format Namespace.Validator_0

  create_key <name>                    Create a new key
    Parameters:
      name                             Name for the new key

  transfer_balance                     Transfer balance between keys
    Parameters:
      source_key                       Source key name
      target_key                       Target key name
      amount                           Amount to transfer

  register_miner <name>                Register a miner
    Parameters:
      name                             Miner name in format Namespace.Miner_0
      --port <port>                    Optional: Specify port (default: auto-assigned)
      --provider <provider>            Optional: Specify provider (anthropic/openrouter)

  register_validator <name>            Register a validator
    Parameters:
      name                             Validator name in format Namespace.Validator_0
      --port <port>                    Optional: Specify port (default: auto-assigned)

  update_module <name>                 Update a module
    Parameters:
      name                             Module name to update

  deploy_miner <name>                  Deploy a miner
    Parameters:
      name                             Miner name to deploy

  deploy_validator <name>              Deploy a validator
    Parameters:
      name                             Validator name to deploy

  configure_port_range                 Configure the port range for modules
    Parameters:
      start_port                       Starting port number
      end_port                         Ending port number

  test_miner <name>                    Test a miner's functionality
    Parameters:
      name                             Miner name to test
      --prompt <prompt>                Optional: Test prompt

Global Options:
  --help                              Show this help message
  --setup                             Run initial setup

Environment Variables:
  MODULE_KEYNAME                      Pre-set module name (optional)
  CONFIG_IP_LIMITER_BUCKET_SIZE      Request bucket size for rate limiting
  CONFIG_IP_LIMITER_REFILL_RATE      Rate limit refill rate

Examples:
  ./launch.sh serve_miner OpenAI.Miner_0 --test-mode    # Start a miner with test mode
  ./launch.sh register_miner Anthropic.Miner_0 --port 8080 --provider anthropic
  ./launch.sh serve_validator Text.Validator_0          # Start a validator
  ./launch.sh                                          # Show interactive menu

Port Management:
  - Ports are stored in ~/.commune/miner_ports.txt
  - Each module needs a consistent port across registrations and serving
  - Default port range: 8000-9000

Notes:
  - Key names should be unique across your deployment
  - Provider selection affects which API will be used
  - Test mode increases rate limits for development
  - Always ensure proper configuration in env/config.env
EOF
}

print_menu() {
    clear
    echo "=== Synthia Deployment Menu ==="
    echo ""
    echo "Deployment Operations:"
    echo "  1. Deploy Validator - serve and launch"
    echo "  2. Deploy Miner - serve and launch"
    echo "  3. Deploy Both - serve and launch validator and miner"
    echo ""
    echo "Individual Operations:"
    echo "  4. Register Validator"
    echo "  5. Register Miner"
    echo "  6. Serve Validator"
    echo "  7. Serve Miner"
    echo ""
    echo "Module Operations:"
    echo "  8. Update Module"
    echo "  9. Configure Port Range"
    echo ""
    echo "Balance Operations:"
    echo "  10. Transfer Balance"
    echo "  11. Unstake and Transfer Balance"
    echo "  12. Unstake and Transfer Balance - Multiple"
    echo "  13. Unstake and Transfer Balance - All"
    echo "  14. Unstake and Transfer Balance - By Name"
    echo "  15. Transfer and Stake Multiple"
    echo ""
    echo "Testing & Management:"
    echo "  16. Create Key"
    echo "  17. Test Miner"
    echo "  18. Exit"
    echo ""
    echo "To activate the Python environment again after exiting, run:"
    echo "source .venv/bin/activate"
    echo ""
}

main_menu() {
    while true; do
        print_menu
        read -r -p "Choose an option (1-18): " choice
        
        case $choice in
            1) deploy_validator ;;
            2) deploy_miner ;;
            3)
                deploy_validator
                deploy_miner
                ;;
            4) register_validator ;;
            5) register_miner ;;
            6) serve_validator ;;
            7) serve_miner ;;
            8) update_module ;;
            9) configure_port_range ;;
            10) transfer_balance ;;
            11) unstake_and_transfer ;;
            12) unstake_and_transfer_multiple ;;
            13) unstake_and_transfer_all ;;
            14) unstake_and_transfer_by_name ;;
            15) transfer_and_stake_multiple ;;
            16) create_key ;;
            17) test_miner ;;
            18) 
                echo "Exiting menu..."
                echo "To reactivate the Python environment, run: source .venv/bin/activate"
                exit 0
                ;;
            *)
                echo "Invalid option. Please try again."
                ;;
        esac
        
        echo ""
        read -r -p "Press Enter to continue..."
    done
}

# Main script execution
if [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

if [ "$1" = "--setup" ]; then
    setup_environment
    exit 0
fi

# Ensure we're in virtual environment and show menu
ensure_venv "$1"
main_menu

# After menu exits, start an interactive shell to keep the environment active
if [ -n "$VIRTUAL_ENV" ]; then
    # Start a new interactive shell that inherits our environment
    exec "$SHELL" -i
fi