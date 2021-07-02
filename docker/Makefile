TAG_NAME=$(USER)/ariel

# constrain the build node: make BUILD_ARGS="--build-arg constraint:node==s876cn03"
BUILD_ARGS=

# push to $DOCKER_REPOSITORY, but only if this variable is set
push: .PUSH
.PUSH: .IMAGE
ifndef DOCKER_REPOSITORY
	$(error $$DOCKER_REPOSITORY is not set, run 'make image' to build locally)
else
	- docker rmi -f $(DOCKER_REPOSITORY)/$(TAG_NAME)
	docker tag $(TAG_NAME) $(DOCKER_REPOSITORY)/$(TAG_NAME)
	docker push $(DOCKER_REPOSITORY)/$(TAG_NAME)
	docker pull $(DOCKER_REPOSITORY)/$(TAG_NAME)
	echo "$(DOCKER_REPOSITORY)/$(TAG_NAME)" > $@
endif

# build the image with a local tag
image: .IMAGE
.IMAGE: Dockerfile
	- docker rmi -f $(TAG_NAME)
	docker build --tag $(TAG_NAME) .
	echo "$(TAG_NAME)" > $@

clean:
	- docker rmi -f $(TAG_NAME)
	- docker rmi -f $(DOCKER_REPOSITORY)/$(TAG_NAME)
	rm -f .IMAGE .PUSH

.PHONY: image push clean
