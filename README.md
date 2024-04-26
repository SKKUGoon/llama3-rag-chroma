# Llama3-Rag

## Installation

### On Windows

Use `wsl` on windows for better developer experience.

Set up `pyenv` and `pyenv-virtualenv`.

```bash
sudo apt-get update

# Install `pyenv`
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
exec "$SHELL"

# Install `pyenv-virtualenv`
sed -Ei -e '/^([^#]|$)/ {a \
export PYENV_ROOT="$HOME/.pyenv"
a \
export PATH="$PYENV_ROOT/bin:$PATH"
a \
' -e ':a' -e '$!{n;ba};}' ~/.profile

echo 'eval "$(pyenv init --path)"' >>~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

Exit the `wsl` and enter again.

```bash
# Test installation
pyenv --version
```

And set up an virtual environment.

```bash
pip install -r requirements.txt
```

