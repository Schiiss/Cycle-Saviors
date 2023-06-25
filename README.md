# :bicyclist: CycleSaviors
Inaugural Pinecone hackathon submission.

[Click to see our deployed submission!!!!](https://huggingface.co/spaces/cycle-saviours/Find_My_Stolen_Bike)

## Identified Problem
It's estimated that in North America bike theft results in more than $500 million in losses each year. It's also estimated that a bike is stolen every 30 seconds and impacts nearly 2 million cyclists<sup>1</sup>. Bikes, unlike other forms of transportation, don't require a registration number, and as such are easy to re-sell in on-line marketplaces such as eBay, Craigslist, Kijiji, and Facebook<sup>1</sup>. 

The best tools to combat bike theft are proactive, these include locks, alarms, keeping your bike inside, anything that can be done to prevent the theft from happening. Once a bike is stolen, there are several reactive tools and services that exist to help locate and recover the bike. These include bike registries, such as [project529](https://project529.com/garage/) and [BikeIndex](https://bikeindex.org/) where stolen bikes can be cross-listed with found bikes, hidden gps trackers, and law enforcement. 

None of the current reactive solutions provide a way to more efficiently locate a stolen bike across the wide array of on-line marketplaces. As such, the process of finding a stolen bike relies heavily on manual search and online bike forums, where users can ask other users to "keep an eye out". 

<sup>1</sup>https://project529.com/garage/org_faq/en/fighting%20bike%20theft/background-on-bike-theft/

## Proposed Solution
We want to help fill the gap in stolen bike searchability by creating a bike theft assistant that will help a victim search for their bike across multiple on-line marketplaces using an image of their bike and textual metadata. If the bike is found, the assistant will provide guidance on actions to take, such as alerting the online marketplace that this is a stolen bike or contacting the lister. 

<strong>This tool differs from current search solutions in that we use vector embeddings of images along with textual metadata to filter and find similar ads listed in online marketplaces based on vector similarity. This is a much more efficient way to find a stolen bike versus keyword search.</strong>

## Proposed Architecture
Conner to update

## Getting Started
Dependency management and environment are handled in `devcontainer`
1. Clone this repository
2. Ideally open the codebase in VS Code
3. Rename `configs/.env-example` to `configs/.env` and fill in the secrets
4. In VS Code run `ctrl+shift+p` to open the command pallete, type `Dev Containers: Rebuild and Reopen in Container` - you are now developing in custom container.
    - All packages will be installed container wide from `requirements.txt` pinned to Python 3.10
    - All functions under `src` will be installed as an editable package and available for import anywhere in repo
    - Keys defined in `configs/.env` will be available container wide from `os.getenv()` - no need to `loaddotenv()` in python
2. You can `pip install` directly in the container. When adding new packages before making a commit run `pip list --format=freeze > requirements.txt --exclude-editable`

#### File Structure
Overview of repository codebase

├── .devcontainer           # Containerized environment management
├── app                     # Frontend application files
├── configs                 # Secrets
├── data                    # Output of batch pipelines
├── pipelines               # Batch pipelines
│   ├── data                # Get data
│   └── index               # Create Pinecone index and upsert
├── src                     # Source files installed in editable mode
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── setup.cfg
└── README.md
