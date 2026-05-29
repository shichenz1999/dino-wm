import hydra
from omegaconf import OmegaConf

@hydra.main(config_path=None)
def register_resolvers(cfg):
    pass

# Define the resolver function
def replace_slash(value: str) -> str:
    return value.replace('/', '_')


def variant_path(variant_type: str, variant: str) -> str:
    """Map (variant_type, variant) to a directory path segment.

    baseline → "baseline"; otherwise → "{variant_type}/{variant}".
    """
    if variant_type == "baseline":
        return "baseline"
    return f"{variant_type}/{variant}"


# Register the resolver with Hydra
OmegaConf.register_new_resolver("replace_slash", replace_slash)
OmegaConf.register_new_resolver("variant_path", variant_path)

if __name__ == "__main__":
    register_resolvers()

