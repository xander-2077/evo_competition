import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../cfg", config_name="config", version_base="1.3")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()