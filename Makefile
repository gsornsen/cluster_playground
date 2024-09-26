.PHONY: conda env clean server setup update

conda:
	@echo "Downloading and installing Miniconda..."
	@wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
	@bash miniconda.sh -b -p $$HOME/miniconda
	@rm miniconda.sh
	@echo "export PATH=$$HOME/miniconda/bin:$$PATH" >> $$HOME/.bashrc
	@echo "Miniconda installed. Please restart your terminal or run 'source $$HOME/.bashrc' to update your PATH."

env:
	$$HOME/miniconda/bin/conda env create -f environment.yml

update:
	$$HOME/miniconda/bin/conda env update -f environment.yml

clean:
	$$HOME/miniconda/bin/conda env remove -n rapids-24.08

setup: env
