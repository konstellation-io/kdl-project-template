# Makefile

# Variables
PROJECT_CONFIG_FILES = lab/processes/config.ini
TEMPLATE_PROJECT_NAME = kdl-project-template
NEW_PROJECT_NAME = $(shell basename $$PWD)

DRONE_FILE = .drone.yml
DRONE_TEMPLATE_PROJECT_NAME = project-template

SONAR_FILE = sonar-project.properties

setup_project:
	@echo "Setting up the project..."
	@sed -si 's/$(TEMPLATE_PROJECT_NAME)/$(NEW_PROJECT_NAME)/g' $(PROJECT_CONFIG_FILES) > /tmp/sed_output; \
	if [ $$? -eq 0 ]; then \
		echo "\033[32m✓ All project references where updated to $(NEW_PROJECT_NAME) in $(PROJECT_CONFIG_FILES)!\033[0m"; \
	else \
		echo "\033[31m× There was an error updating the references in the `$(PROJECT_CONFIG_FILES)` file, please check for all `$(TEMPLATE_PROJECT_NAME)` references and replace them with your project name!\033[0m"; \
	fi

	@sed -si 's/$(DRONE_TEMPLATE_PROJECT_NAME)/$(NEW_PROJECT_NAME)/g' $(DRONE_FILE) > /tmp/sed_output; \
	if [ $$? -eq 0 ]; then \
		echo "\033[32m✓ All project references where updated to $(NEW_PROJECT_NAME) in $(DRONE_FILE)!\033[0m"; \
	else \
		echo "\033[31m× There was an error updating the references in the `$(PROJECT_CONFIG_FILES)` file, please check for all `$(TEMPLATE_PROJECT_NAME)` references and replace them with your project name!\033[0m"; \
	fi

	git add $(PROJECT_CONFIG_FILES) $(DRONE_FILE)
	git commit -m "Setup project"
	
	@echo "Do you want to push the changes to the repository? [Y/n]"
	@read PUSH; \
	if [ "$${PUSH:-Y}" = "y" ] || [ "$${PUSH:-Y}" = "Y" ]; then \
		git push && echo "\033[32m✓ Changes have been successfully pushed to the repository!\033[0m"; \
	else \
		echo "\033[33m⚠ Please remember to push the changes to the repository!\033[0m"; \
	fi

	@echo "\033[33m⚠ Please remember to update the $(SONAR_FILE) file with your project name and key, and rename the project name in the README file!\033[0m"
