# Makefile for DeepH-pack documentation
# Usage: make [target]

.PHONY: help install build livehtml clean gen-cap deploy

# Variables
PORT ?= 8082
HOST ?= 0.0.0.0
SPHINXOPTS ?=
SPHINXAUTOBUILD ?= sphinx-autobuild
SPHINXBUILD ?= sphinx-build
BUILDDIR = _build
SOURCEDIR = .

# Default target
help:
	@echo "DeepH-dock Documentation Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies for building docs"
	@echo "  gen-cap     - Generate capabilities documentation"
	@echo "  build       - Build documentation (static)"
	@echo "  livehtml    - Build and serve with live reload"
	@echo "  clean       - Remove build directory"
	@echo "  serve       - Serve built documentation (no auto-rebuild)"
	@echo "  deploy      - Build for deployment (clean + build)"
	@echo ""
	@echo "Variables:"
	@echo "  PORT=port   - Port for livehtml/serve (default: 8082)"
	@echo "  HOST=host   - Host for livehtml/serve (default: 0.0.0.0)"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make build"
	@echo "  make livehtml PORT=8082"
	@echo "  make deploy"

# Install dependencies
install:
	@echo "Installing documentation dependencies..."
	uv pip install ..[docs]
	@echo "Done!"

# Generate capabilities
gen-cap:
	@echo "Generating capabilities documentation..."
	python gen_capabilities.py
	@echo "Capabilities generated!"

# Build static documentation
build: gen-cap
	@echo "Building static documentation..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Build finished. HTML pages are in $(BUILDDIR)/html/"

# Build and serve with live reload
livehtml: clean gen-cap
	@echo "Starting live documentation server on http://$(HOST):$(PORT)"
	@echo "Press Ctrl+C to stop"
	$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--host "$(HOST)" \
		--port "$(PORT)" \
		--watch "$(SOURCEDIR)" \
		--re-ignore "\.#.*" \
		$(SPHINXOPTS)

# Serve built documentation without auto-rebuild
serve:
	@echo "Serving built documentation on http://$(HOST):$(PORT)"
	cd "$(BUILDDIR)/html" && python -m http.server "$(PORT)" --bind "$(HOST)"

# Clean build directory
clean:
	@echo "Cleaning build directory..."
	rm -rf "$(BUILDDIR)"
	@echo "Cleaned!"

# Build for deployment
deploy: clean build
	@echo "Documentation built for deployment in $(BUILDDIR)/html/"
	@echo "Ready to deploy!"

# Quick check of documentation syntax
check:
	@echo "Checking documentation syntax..."
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)
	@echo "Link check complete. Report in $(BUILDDIR)/linkcheck/output.txt"
