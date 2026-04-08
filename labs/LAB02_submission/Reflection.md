# Lab 2 Reflection

In this lab, both containers ran on your laptop. In production, the preprocessor would run in the warehouse datacenter and the inference API would run in Congo's main datacenter.

**How would the architecture and your `docker run` commands differ if these containers were actually running in separate datacenters?**

Consider:
- How would the preprocessor find the inference API?
- What about the shared volumes?
- What new challenges would arise?


## Your Reflection Below

Running both containers on my laptop made networking easy. I could connect them using host.docker.internal. In a real world setup with two datacenters, this would not work. The preprocessor would have to reach the inference API over the internet. That means the API would need a real domain name or a static IP address, and the API_URL environment variable would need to point to it. Additionally, some form of authentication would also be necessary so random users could not access the /predict endpoint.

Taking into account the two different data centers, this setup would be complicated. Currently, both containers read/write from and to folders on my computer. In separate datacenters, the incoming and logs folders would need to be replaced with something like Google Cloud Storage that both locations can access on their own. Viewing this in a grand scale, we will need to take into account things such as latency and reliability as the connection between the warehouse and the main datacenter becomes a big concern. This is something that we did not have to account for as we were simply reading and writing onto the same device.