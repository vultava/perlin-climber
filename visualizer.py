import numpy as np
import torch
import environment
from agent import DQNAgent
import cv2

def viz_trained_agent(model_path: str = "perlin_climber_dqn_64_withEuclid_v2_at5000.pth", map_dim: int = 128):

    map = environment.PerlinMap(np.zeros((map_dim, map_dim)), map_dim, 0, 0, 0, 0, 0, 500)
    map.create_map(4, 2, 0.5)
    state = map.reset()

    #map.climber_x, map.climber_y = 0, 20
    #map.final_x, map.final_y = 100, 100
    #state = map.get_state()
    print(f"climber_x {map.climber_x} | climber_y {map.climber_y} | final_x {map.final_x} | final_y {map.final_y}")

    agent = DQNAgent(6, 4)
    agent.policy_nn.load_state_dict(torch.load(model_path))
    agent.policy_nn.eval()

    path_x, path_y = [map.climber_x], [map.climber_y]
    done = False

    while not done:
        action = agent.select_action(state, 0.0)
        state, rew, done, _ = map.step(action)
        path_x.append(map.climber_x)
        path_y.append(map.climber_y)


    # IMAGE FOMRATION
    map_img = np.zeros((map_dim, map_dim, 3), dtype=np.uint8)
    map_img[:,:,0] = (map.map * 255).astype(np.uint8)
    map_img[:,:,1] = (map.map * 255).astype(np.uint8)
    map_img[:,:,2] = (map.map * 255).astype(np.uint8)

    map_img[path_y, path_x, :] = 255, 0, 0
    map_img[path_y[0], path_x[0], :] = 0, 255, 0
    map_img[map.final_y, map.final_x, :] = 0, 0, 255

    win_name = 'Perlin Climber'
    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(win_name, map_dim*5, map_dim*5)
    cv2.imshow(win_name, map_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()



if __name__ == "__main__":
    viz_trained_agent()
