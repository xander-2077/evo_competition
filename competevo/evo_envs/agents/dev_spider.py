from gym_compete.new_envs.agents.agent import Agent
from gymnasium.spaces import Box
import numpy as np
import os

from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO

SCALE_MAX = 1.2 #0.5

class DevSpider(Agent):

    def __init__(self, agent_id, cfg, xml_path=None, n_agents=2):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "spider_body.xml")
        super(DevSpider, self).__init__(agent_id, xml_path, n_agents)

        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(xml_path, parser=parser)
        self.cur_xml_str = etree.tostring(self.tree, pretty_print=True).decode('utf-8')

        self.cfg = cfg

        self.scale_vector = np.random.uniform(low=-1., high=1., size=40)
    
    @property
    def flag(self):
        return "dev"
    
    def set_env(self, env):
        self.env = env
        self._env_init = True
        self._set_body()
        self._set_joint()
        if self.n_agents > 1:
            self._set_other_joint()
        self.set_observation_space()
        self.set_action_space()

        # dimension definition
        self.scale_state_dim = self.scale_vector.size
        self.sim_obs_dim = self.observation_space.shape[0]
        self.sim_action_dim = self.action_space.shape[0]

    def set_design_params(self, action):
        scale_state = action[:self.scale_state_dim]
        self.scale_vector = scale_state
        # print(scale_state)

        design_params = self.scale_vector * SCALE_MAX
        a = design_params + 1.
        b = design_params*0.5 + 1 # for gear only

        def multiply_str(s, m):
            res = [str(float(x) * m) for x in s.split()]
            res_str = ' '.join(res)
            return res_str

        agent_body = self.tree.find('body')
        for body in agent_body.iter('body'):
            cur_name = body.get('name')

            # 1
            if cur_name == "1":
                geom = body.find('geom') #1
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[0])
                    geom.set("fromto", p)

            if cur_name == "11":
                p = body.get("pos")
                p = multiply_str(p, a[0])
                body.set("pos", p)

                geom = body.find('geom') #11
                p = geom.get("size")
                p = multiply_str(p, a[1])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[2])
                    geom.set("fromto", p)

            if cur_name == "111":
                p = body.get("pos")
                p = multiply_str(p, a[2])
                body.set("pos", p)

                geom = body.find('geom') #111
                p = geom.get("size")
                p = multiply_str(p, a[3])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[4])
                    geom.set("fromto", p)

            # 2
            if cur_name == "2":
                geom = body.find('geom') #2
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[5])
                    geom.set("fromto", p)

            if cur_name == "12":
                p = body.get("pos")
                p = multiply_str(p, a[5])
                body.set("pos", p)

                geom = body.find('geom') #12
                p = geom.get("size")
                p = multiply_str(p, a[6])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[7])
                    geom.set("fromto", p)

            if cur_name == "112":
                p = body.get("pos")
                p = multiply_str(p, a[7])
                body.set("pos", p)

                geom = body.find('geom') #112
                p = geom.get("size")
                p = multiply_str(p, a[8])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[9])
                    geom.set("fromto", p)

            # 3
            if cur_name == "3":
                geom = body.find('geom') #3
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[10])
                    geom.set("fromto", p)

            if cur_name == "13":
                p = body.get("pos")
                p = multiply_str(p, a[10])
                body.set("pos", p)

                geom = body.find('geom') #13
                p = geom.get("size")
                p = multiply_str(p, a[11])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[12])
                    geom.set("fromto", p)

            if cur_name == "113":
                p = body.get("pos")
                p = multiply_str(p, a[12])
                body.set("pos", p)

                geom = body.find('geom') #113
                p = geom.get("size")
                p = multiply_str(p, a[13])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[14])
                    geom.set("fromto", p)

            # 4
            if cur_name == "4":
                geom = body.find('geom') #4
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[15])
                    geom.set("fromto", p)

            if cur_name == "14":
                p = body.get("pos")
                p = multiply_str(p, a[15])
                body.set("pos", p)

                geom = body.find('geom') #14
                p = geom.get("size")
                p = multiply_str(p, a[16])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[17])
                    geom.set("fromto", p)

            if cur_name == "114":
                p = body.get("pos")
                p = multiply_str(p, a[17])
                body.set("pos", p)

                geom = body.find('geom') #114
                p = geom.get("size")
                p = multiply_str(p, a[18])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[19])
                    geom.set("fromto", p)

            # 5
            if cur_name == "5":
                geom = body.find('geom') #5
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[20])
                    geom.set("fromto", p)

            if cur_name == "15":
                p = body.get("pos")
                p = multiply_str(p, a[20])
                body.set("pos", p)

                geom = body.find('geom') #15
                p = geom.get("size")
                p = multiply_str(p, a[21])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[22])
                    geom.set("fromto", p)

            if cur_name == "115":
                p = body.get("pos")
                p = multiply_str(p, a[22])
                body.set("pos", p)

                geom = body.find('geom') #115
                p = geom.get("size")
                p = multiply_str(p, a[23])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[24])
                    geom.set("fromto", p)
            
            # 6
            if cur_name == "6":
                geom = body.find('geom') #6
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[25])
                    geom.set("fromto", p)

            if cur_name == "16":
                p = body.get("pos")
                p = multiply_str(p, a[25])
                body.set("pos", p)

                geom = body.find('geom') #16
                p = geom.get("size")
                p = multiply_str(p, a[26])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[27])
                    geom.set("fromto", p)

            if cur_name == "116":
                p = body.get("pos")
                p = multiply_str(p, a[27])
                body.set("pos", p)

                geom = body.find('geom') #116
                p = geom.get("size")
                p = multiply_str(p, a[28])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[29])
                    geom.set("fromto", p)

            # 7
            if cur_name == "7":
                geom = body.find('geom') #7
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[30])
                    geom.set("fromto", p)

            if cur_name == "17":
                p = body.get("pos")
                p = multiply_str(p, a[30])
                body.set("pos", p)

                geom = body.find('geom') #17
                p = geom.get("size")
                p = multiply_str(p, a[31])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[32])
                    geom.set("fromto", p)

            if cur_name == "117":
                p = body.get("pos")
                p = multiply_str(p, a[32])
                body.set("pos", p)

                geom = body.find('geom') #117
                p = geom.get("size")
                p = multiply_str(p, a[33])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[34])
                    geom.set("fromto", p)
            
            # 8
            if cur_name == "8":
                geom = body.find('geom') #8
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[35])
                    geom.set("fromto", p)

            if cur_name == "18":
                p = body.get("pos")
                p = multiply_str(p, a[35])
                body.set("pos", p)

                geom = body.find('geom') #18
                p = geom.get("size")
                p = multiply_str(p, a[36])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[37])
                    geom.set("fromto", p)

            if cur_name == "118":
                p = body.get("pos")
                p = multiply_str(p, a[37])
                body.set("pos", p)

                geom = body.find('geom') #118
                p = geom.get("size")
                p = multiply_str(p, a[38])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[39])
                    geom.set("fromto", p)

        agent_actuator = self.tree.find('actuator')
        for motor in agent_actuator.iter("motor"):
            cur_name = motor.get("name").split('_')[0]

            if cur_name == "11":
                p = motor.get("gear")
                p = multiply_str(p, b[1])
                motor.set("gear", p)

            if cur_name == "111":
                p = motor.get("gear")
                p = multiply_str(p, b[3])
                motor.set("gear", p)

            if cur_name == "12":
                p = motor.get("gear")
                p = multiply_str(p, b[6])
                motor.set("gear", p)

            if cur_name == "112":
                p = motor.get("gear")
                p = multiply_str(p, b[8])
                motor.set("gear", p)

            if cur_name == "13":
                p = motor.get("gear")
                p = multiply_str(p, b[11])
                motor.set("gear", p)

            if cur_name == "113":
                p = motor.get("gear")
                p = multiply_str(p, b[13])
                motor.set("gear", p)

            if cur_name == "14":
                p = motor.get("gear")
                p = multiply_str(p, b[16])
                motor.set("gear", p)

            if cur_name == "114":
                p = motor.get("gear")
                p = multiply_str(p, b[18])
                motor.set("gear", p)
            
            if cur_name == "15":
                p = motor.get("gear")
                p = multiply_str(p, b[21])
                motor.set("gear", p)

            if cur_name == "115":
                p = motor.get("gear")
                p = multiply_str(p, b[23])
                motor.set("gear", p)

            if cur_name == "16":
                p = motor.get("gear")
                p = multiply_str(p, b[26])
                motor.set("gear", p)

            if cur_name == "116":
                p = motor.get("gear")
                p = multiply_str(p, b[28])
                motor.set("gear", p)

            if cur_name == "17":
                p = motor.get("gear")
                p = multiply_str(p, b[31])
                motor.set("gear", p)

            if cur_name == "117":
                p = motor.get("gear")
                p = multiply_str(p, b[33])
                motor.set("gear", p)

            if cur_name == "18":
                p = motor.get("gear")
                p = multiply_str(p, b[36])
                motor.set("gear", p)

            if cur_name == "118":
                p = motor.get("gear")
                p = multiply_str(p, b[38])
                motor.set("gear", p)

        # print(etree.tostring(self.tree, pretty_print=True).decode('utf-8'))
        self.cur_xml_str = etree.tostring(self.tree, pretty_print=True).decode('utf-8')
        # print(self.cur_xml_str)

    def set_goal(self, goal):
        self.GOAL = goal
        self.move_left = False
        if self.get_qpos()[0] > 0:
            self.move_left = True

    def before_step(self):
        self._xposbefore = self.get_body_com("0")[0]

    def after_step(self, action):
        xposafter = self.get_body_com("0")[0]
        forward_reward = (xposafter - self._xposbefore) / self.env.dt
        if self.move_left:
            forward_reward *= -1
        ctrl_cost = .08 * np.square(action).sum()
        cfrc_ext = self.get_cfrc_ext()
        contact_cost = 0

        qpos = self.get_qpos()
        agent_standing = qpos[2] >= 0.3 and qpos[2] <= 1.2
        survive = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive

        reward_info = dict()
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_dense'] = reward

        terminated = not agent_standing
        
        return reward, terminated, reward_info

    def _get_obs(self):
        '''
        Return agent's observations
        '''
        raise NotImplementedError("Error! This function should be designed by you!")

    def set_observation_space(self):
        obs = self._get_obs()
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)

    def reached_goal(self):
        if self.n_agents == 1: return False
        xpos = self.get_body_com('0')[0]
        if self.GOAL > 0 and xpos > self.GOAL:
            return True
        elif self.GOAL < 0 and xpos < self.GOAL:
            return True
        return False

    def reset_agent(self):
        xpos = self.get_qpos()[0]
        if xpos * self.GOAL > 0 :
            self.set_goal(-self.GOAL)
        if xpos > 0:
            self.move_left = True
        else:
            self.move_left = False
        
        self.scale_vector = np.random.uniform(low=-1., high=1., size=40)
