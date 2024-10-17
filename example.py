from PIL import Image
import depth_pro

class Model:
    # Singleton model

    def __init__(self):
        # Initialize the model.
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()

    def load_image(self, image_path: str):
        image, _, f_px = depth_pro.load_rgb(image_path)
        self.transformed_image = self.transform(image)
        self.f_px = f_px
        self.image_path = image_path
        self.image = image
    
    def infer(self):
        prediction = self.model.infer(self.transformed_image, f_px=self.f_px)
        self.depth = prediction["depth"]
        self.focallength_px = prediction["focallength_px"]
        return prediction["depth"], prediction["focallength_px"]
    
    def load_and_infer(self, image_path: str):
        """Convenience method to load an image and infer depth."""
        self.load_image(image_path)
        return self.infer()

if __name__=="__main__":
    model = Model()
    # image_path = "./data/my_photo.jpg"
    # image_path = "./data/mom_still_walking.jpg"
    image_path = "./data/pool_balls.jpg"
    model.load_image(image_path)
    depth, f_px = model.infer()
    print(depth)
