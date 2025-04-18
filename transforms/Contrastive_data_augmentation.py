import torch
from sign_language_tools.pose.transform.rotate import Rotation2D
from sign_language_tools.pose.transform.translation import Translation
from sign_language_tools.pose.transform import noise
from sign_language_tools.pose.transform import GaussianNoise
from sign_language_tools.pose.transform import flip
from sign_language_tools.pose.transform import smooth
from sympy import rotations


class DataAugmentation:
    def __init__(self):
        pass

    class Rotate:
        def __init__(self, angle):
            self.rotation = Rotation2D(angle)
        def __call__(self, left_hand, rigth_hand, pose):
            left_hand_rot = [self.rotation(left_hand[i].cpu().numpy()) for i in range(left_hand.shape[0])]
            rigth_hand_rot = [self.rotation(rigth_hand[i].cpu().numpy()) for i in range(rigth_hand.shape[0])]
            pose_rot = [self.rotation(pose[i].cpu().numpy()) for i in range(pose.shape[0])]
            return torch.tensor(left_hand_rot).to('cuda'), torch.tensor(rigth_hand_rot).to('cuda'), torch.tensor(pose_rot).to('cuda')

    class Translate:
        def __init__(self, abs, ord):
            self.translation = Translation(abs, ord)

        def __call__(self, left_hand, rigth_hand, pose):
            left_hand_rot = [self.translation(left_hand[i].cpu().numpy()) for i in range(left_hand.shape[0])]
            rigth_hand_rot = [self.translation(rigth_hand[i].cpu().numpy()) for i in range(rigth_hand.shape[0])]
            pose_rot = [self.translation(pose[i].cpu().numpy()) for i in range(pose.shape[0])]
            return torch.tensor(left_hand_rot).to('cuda'), torch.tensor(rigth_hand_rot).to('cuda'), torch.tensor(pose_rot).to('cuda')

    class GaussianNoise:
        def __init__(self, std):
            self.noise = GaussianNoise(std) 

        def __call__(self, left_hand, right_hand, pose):
            left_hand_noisy = [self.noise(left_hand[i].cpu().numpy()) for i in range(left_hand.shape[0])]
            right_hand_noisy = [self.noise(right_hand[i].cpu().numpy()) for i in range(right_hand.shape[0])]
            pose_noisy = [self.noise(pose[i].cpu().numpy()) for i in range(pose.shape[0])]
            return torch.tensor(left_hand_noisy).to('cuda'), torch.tensor(right_hand_noisy).to('cuda'), torch.tensor(pose_noisy).to('cuda')

    class HorizontalFlip:
        def __call__(self, left_hand, rigth_hand, pose):
            left_hand = flip.HorizontalFlip(left_hand.squeeze(0).cpu().numpy())
            rigth_hand = flip.HorizontalFlip(rigth_hand.squeeze(0).cpu().numpy())
            pose = flip.HorizontalFlip(pose.squeeze(0).cpu().numpy())
            return torch.tensor(left_hand), torch.tensor(rigth_hand), torch.tensor(pose)

    class VFlip:
        def __call__(self, landmarks: torch.Tensor):
            shape = landmarks.shape
            flipped = torch.stack([self.flip(l) for l in landmarks])
            return flipped.to(landmarks.device)

        def flip(self, landmark: torch.Tensor):
            landmark[..., 1] = 1 - landmark[..., 1]
            return landmark
        def __call__(self, left_hand, rigth_hand, pose):
            left_hand = self.flip(left_hand.squeeze(0).numpy())
            rigth_hand = self.flip(rigth_hand.squeeze(0).numpy())
            pose = self.flip(pose.squeeze(0).numpy())
            return pose, left_hand, rigth_hand